# Fase 2 — Prototipo de validación: puente por subproceso para backends Python

**Rama:** `188-spike…subprocess-bridge` · **Estado:** prototipo, NO integrado a
`main`, NO es código de producción. Valida la decisión ya tomada (puente por
subproceso, PyO3 embebido descartado) antes de comprometerla en el contrato
(Fase 3) y el registro (Fase 4). No compara opciones.

## Qué es

Un puente mínimo Rust ↔ worker Python de prueba (sin SDK real). Protocolo fijo:

```
frame = u32 little-endian (longitud)  ||  payload JSON UTF-8
```

sobre `stdin`/`stdout` del hijo. Se eligió **JSON** (no msgpack) para el
prototipo: frames legibles hacen un spike mucho más fácil de depurar, y la
medición de overhead (punto 4) demuestra que el formato **no** es el cuello de
botella. Piezas:

- `src/lib.rs` — el puente: `Worker::spawn`, `send`/`recv`/`call`, `signal`,
  y el enum `BridgeError` (materia prima del tipo de error de la Fase 3).
- `python/worker.py` — worker trivial: responde `ping`, ejecuta `run`
  (conteos falsos deterministas), y **captura SIGINT/SIGTERM** para abortar una
  llamada en curso y quedar reutilizable.
- `tests/` — un test por riesgo, ejecutando subprocesos de verdad.
- `src/bin/overhead.rs` — mide el round-trip real del protocolo.
- `slurm/orphan_job.sbatch` + `src/bin/orphan_probe.rs` — prueba de huérfanos
  bajo SLURM.

## Cómo reproducir

```bash
cd spikes/subprocess-bridge
export SPIKE_PYTHON=$(which python3)
cargo test --tests                     # puntos 1, 2a, 2b
cargo run --release --bin overhead     # punto 2d
# punto 2c (SLURM):
export SPIKE_DIR=$(pwd)
sbatch --export=ALL,SPIKE_DIR,SPIKE_PYTHON slurm/orphan_job.sbatch
```

Es un **workspace de Cargo independiente** (tabla `[workspace]` vacía en
`Cargo.toml`): no es miembro del workspace de Polypus, no lo construye
`cargo … --workspace`, y no arrastra ninguna dependencia nuestra.

---

## Resultado de los cuatro riesgos

### 1. Happy path (punto 1) — ✅

`tests/happy_path.rs`: Rust lanza el worker, manda `run` con 2 circuitos y 1024
shots, recibe un histograma de conteos por circuito, y reutiliza el mismo
proceso para una segunda llamada. Verde.

### 2a. Cancelación — ✅ con requisito de protocolo

`tests/cancellation.rs`. Con una llamada de 2 s en curso, `kill(SIGINT)` desde
Rust:

- El worker **se entera por la señal** (handler SIGINT/SIGTERM), aborta el sleep
  troceado en curso en <1,5 s, responde `{"op":"aborted"}` y **sigue vivo y
  reutilizable** (un `ping` posterior devuelve `pong`).
- `SIGKILL` **no** es capturable → colapsa en el camino de caída (2b).

**Hallazgo que fuerza el protocolo (no bloqueante):** lo único que alcanza a un
worker ocupado dentro de una llamada de cómputo/QPU es **una señal del SO**.
Cerrar el pipe no cancela nada, porque el worker no está leyendo el pipe
mientras trabaja. Por tanto:

- la primitiva de cancelación del puente debe ser "señalar al hijo"
  (SIGINT/SIGTERM), no "cerrar la conexión";
- el protocolo **necesita una respuesta `aborted` explícita** para que un worker
  sobreviva a la cancelación y se reutilice (si no, cancelar = matar y respawn).

Esto encaja con el `CancelToken` del Scheduler (#158): el token, al dispararse,
debe traducirse a una señal al subproceso.

### 2b. Caída del subproceso a mitad de llamada — ✅

`tests/crash.rs`, dos vías:

- **SIGKILL externo** con petición pendiente: Rust detecta EOF en el pipe y
  devuelve `BridgeError::WorkerDied { wait_status: "killed by signal 9" }` — un
  **error tipado explícito, sin cuelgue y sin pánico**. Es **recuperable**: un
  worker nuevo atiende con normalidad.
- **Auto-crash** del worker (`os._exit(137)` a mitad de `run`, imitando un
  segfault/abort del SDK): mismo resultado, `WorkerDied`.

`WorkerDied` es exactamente la variante `External` / "el backend dejó de
responder" que definirá la Fase 3. Lleva el `wait_status` (código vs señal), que
informa si merece la pena reintentar con un worker nuevo.

### 2c. Comportamiento bajo SLURM — ✅ (probado de verdad)

Entorno real: `bahia189` es a la vez host de envío y único nodo de cómputo;
`ProctrackType=proctrack/cgroup`, `TaskPlugin=task/cgroup`.

- **Lanzar el subproceso dentro del job funciona**: el worker vive en el mismo
  cgroup del step (`…/job_380/step_batch/…`), compartiendo la asignación. No hubo
  denegación de fork ni tope de PIDs con `--cpus-per-task=1`.
- **No queda huérfano**: se lanzó un worker huérfano deliberado (sin guarda, con
  `mem::forget` del handle para saltarnos nuestro Drop). `proctrack/cgroup` lo
  mató **al terminar el step**, antes incluso del fin del job. Comprobado desde
  el host tras el teardown del cgroup: el PID ya no existe. **Cero fugas bajo
  SLURM.**
- **Contraste sin SLURM** (`orphan_probe`): un huérfano sin guarda **sí se fuga**
  (reparented a PID 1). Nuestra guarda `PR_SET_PDEATHSIG` lo mata al instante en
  que muere el padre, **sin necesidad de SLURM** — es el respaldo portable para
  ejecución fuera de un job.

**Requisito de sitio (documentar en Fase 4):** este SLURM exige `--cpus-per-task`
y `--mem` explícitos. El subproceso Python **comparte** ese `--mem` y esos cores
con el proceso Rust; el SDK del proveedor + su intérprete deben caber en la
asignación del job. Recomendación: la productivización (Fase 4) debe armar
`PR_SET_PDEATHSIG` **siempre** (cinturón y tirantes: cubre el caso fuera de SLURM
y el de un padre que muere sin ordenar el cierre).

### 2d. Overhead de IPC real — ✅ despreciable

`cargo run --release --bin overhead`, 5000 iteraciones tras warm-up:

| Medición                     | media   | p50     | p99     |
|------------------------------|---------|---------|---------|
| ping/pong round-trip         | ~9 µs   | ~7 µs   | ~25 µs  |
| run(8 circuitos, 0 ms QPU)   | ~16 µs  | ~15 µs  | ~21 µs  |

El round-trip completo del protocolo (serializar + escribir + Python + leer +
deserializar) son **microsegundos**. Frente a una latencia de QPU realista
(cientos de ms a segundos):

- **0,015 %** de una llamada optimista de 100 ms;
- **0,002 %** de una llamada típica de 1 s.

El IPC no es el cuello de botella ni de lejos, y confirma que **JSON basta**:
msgpack ahorraría microsegundos sobre algo ya despreciable. (El coste real de un
backend será dominado por la QPU y, en su caso, por la serialización de payloads
grandes de resultados — no por el ida y vuelta del protocolo.)

---

## Punto 3 — ¿algo bloqueante?

**No hay ningún bloqueante.** El diseño elegido (puente por subproceso) se
sostiene en los cuatro frentes. Dos hallazgos **ajustan el protocolo** (no lo
invalidan) y deben entrar en el diseño de la Fase 3/4:

1. **Cancelación = señal + respuesta `aborted`.** No basta con cerrar el pipe; el
   protocolo debe incluir una respuesta de aborto y la cancelación se implementa
   señalando al hijo. *(Ajuste de protocolo, ya reflejado en el prototipo.)*

2. **Detección de "no responde" necesita algo más que EOF.** El prototipo detecta
   la **muerte** del worker (EOF → `WorkerDied`), que era lo pedido en el punto
   2b. Pero un worker **vivo pero colgado** (deadlock del SDK, QPU que nunca
   contesta) dejaría a `recv()` bloqueado para siempre — este spike no tiene
   timeout. La Fase 4 **debe** añadir un timeout de lectura / heartbeat para
   convertir "colgado" también en la variante "el backend dejó de responder".
   Riesgo abierto, no bloqueante, pero de implementación obligatoria al
   productivizar.

### Riesgos abiertos para Fase 4 (registro / productivización)

- **Timeout/liveness** (arriba): imprescindible.
- **`PR_SET_PDEATHSIG` es Linux-only.** Suficiente para el CESGA (todo Linux),
  pero el crate del puente debe documentarlo o dar alternativa portable.
- **Reparto de recursos SLURM:** el worker comparte `--mem`/cores del job;
  documentar el dimensionamiento y exigir `--cpus-per-task`/`--mem`.
- **Payloads grandes de resultados:** medir con conteos reales grandes (muchos
  shots × muchos circuitos); el overhead medido aquí es con payload mínimo.
