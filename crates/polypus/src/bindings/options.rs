//! The `options` argument of the entry points, converted at the FFI edge.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::infrastructure::OptionValue;

/// The `options` kwarg of the entry points, converted at the FFI edge: a `dict` of
/// `str` keys, each mapped to a `str` or a `list`/`tuple` of `str`
/// ([`OptionValue`]). Anything else — an `int`, `bool`, `float`, `None`, a `dict`, a
/// list holding a non-`str` — is a `TypeError` naming the key (and the position, for
/// a list element), so a mistyped value never reaches a backend factory.
///
/// A local newtype because `FromPyObject` and `OptionValue` are both foreign to this
/// crate. An empty list is accepted here; whether it is meaningful is the backend's
/// call (the subprocess bridge rejects an empty `command`).
pub(crate) struct OptionsArg(HashMap<String, OptionValue>);

impl OptionsArg {
    /// The converted options, as the backend config carries them.
    pub(super) fn into_inner(self) -> HashMap<String, OptionValue> {
        self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for OptionsArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::exceptions::PyTypeError;
        use pyo3::types::{PyString, PyTuple};

        // For error messages only, so an unreadable name is not itself an error.
        let type_name = |item: &Bound<'_, PyAny>| {
            item.get_type()
                .name()
                .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string())
        };
        let dict = ob.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "options must be a dict of str keys to str or list[str] values, got {}",
                type_name(&ob)
            ))
        })?;
        let mut options = HashMap::with_capacity(dict.len());
        // Walk a snapshot of the items, and read each list/tuple by index rather than
        // through `__iter__`: a subclass's `__iter__` is user code that could mutate
        // the dict mid-walk, and PyO3's dict iterator panics when that happens.
        for item in dict.items().iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            let key = match key.cast::<PyString>() {
                Ok(key) => key.to_str()?.to_owned(),
                Err(_) => {
                    return Err(PyTypeError::new_err(format!(
                        "options keys must be str, got {}",
                        type_name(&key)
                    )))
                }
            };
            let value = if let Ok(text) = value.cast::<PyString>() {
                OptionValue::Str(text.to_str()?.to_owned())
            } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
                let elements: Vec<Bound<'py, PyAny>> = match value.cast::<PyList>() {
                    Ok(list) => list.iter().collect(),
                    Err(_) => value.cast::<PyTuple>()?.iter().collect(),
                };
                let mut items = Vec::with_capacity(elements.len());
                for (idx, item) in elements.into_iter().enumerate() {
                    match item.cast::<PyString>() {
                        Ok(text) => items.push(text.to_str()?.to_owned()),
                        Err(_) => {
                            return Err(PyTypeError::new_err(format!(
                                "options['{key}'][{idx}] must be a str, got {}",
                                type_name(&item)
                            )))
                        }
                    }
                }
                OptionValue::List(items)
            } else {
                return Err(PyTypeError::new_err(format!(
                    "options['{key}'] must be a str or a list of str, got {}",
                    type_name(&value)
                )));
            };
            options.insert(key, value);
        }
        Ok(OptionsArg(options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert a Python expression through the `options` kwarg's extractor.
    fn extract_options(expr: &str) -> PyResult<HashMap<String, OptionValue>> {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let code = std::ffi::CString::new(expr).expect("no NUL in the test expression");
            py.eval(&code, None, None)?
                .extract::<OptionsArg>()
                .map(OptionsArg::into_inner)
        })
    }

    #[test]
    fn options_arg_accepts_str_list_and_tuple_values() {
        let options = extract_options(
            "{'endpoint': 'tcp://x:1', 'command': ['python3', '/a b/w.py'], \
              'argv': ('a', 'b c'), 'empty': []}",
        )
        .expect("str, list[str] and tuple[str] values convert");
        assert_eq!(options.len(), 4);
        assert_eq!(options["endpoint"], OptionValue::from("tcp://x:1"));
        assert_eq!(
            options["command"],
            OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()])
        );
        assert_eq!(
            options["argv"],
            OptionValue::from(vec!["a".to_string(), "b c".to_string()])
        );
        // An empty list is accepted at the edge; the backend decides on it.
        assert_eq!(options["empty"], OptionValue::from(Vec::new()));
    }

    /// A `list`/`tuple` subclass whose `__iter__` mutates the `options` dict must not
    /// reach Python code while the dict is walked: PyO3's dict iterator panics when
    /// the dict changes size under it, and a panic must never cross the FFI edge.
    /// The converter reads the dict through a snapshot of its items and the sequences
    /// by index, so `__iter__` never runs and the dict is left untouched.
    #[test]
    fn options_arg_does_not_run_user_iterators_while_walking_the_dict() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let code = std::ffi::CString::new(
                "opts = {}\n\
                 class L(list):\n\
                 \x20   def __iter__(self):\n\
                 \x20       opts['injected-by-list'] = 'x'\n\
                 \x20       return super().__iter__()\n\
                 class T(tuple):\n\
                 \x20   def __iter__(self):\n\
                 \x20       opts['injected-by-tuple'] = 'x'\n\
                 \x20       return super().__iter__()\n\
                 opts['a'] = L(['python3', '/a b/w.py'])\n\
                 opts['b'] = 'y'\n\
                 opts['c'] = T(('t',))\n",
            )
            .expect("no NUL in the test program");
            py.run(&code, Some(&globals), None)
                .expect("the test program runs");
            let opts = globals
                .get_item("opts")
                .expect("globals lookup")
                .expect("the test program defines opts");
            let options = opts
                .extract::<OptionsArg>()
                .map(OptionsArg::into_inner)
                .expect("list/tuple subclasses of str convert");
            assert_eq!(
                options["a"],
                OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()])
            );
            assert_eq!(options["b"], OptionValue::from("y"));
            assert_eq!(options["c"], OptionValue::from(vec!["t".to_string()]));
            assert_eq!(options.len(), 3);
            // The overridden `__iter__` never ran.
            assert_eq!(opts.len().expect("opts is a dict"), 3, "{opts}");
        });
    }

    #[test]
    fn options_arg_rejects_unsupported_values_with_a_type_error_naming_the_key() {
        for (expr, needle) in [
            (
                "{'k': 1}",
                "options['k'] must be a str or a list of str, got int",
            ),
            (
                "{'k': True}",
                "options['k'] must be a str or a list of str, got bool",
            ),
            (
                "{'k': 1.5}",
                "options['k'] must be a str or a list of str, got float",
            ),
            (
                "{'k': None}",
                "options['k'] must be a str or a list of str, got NoneType",
            ),
            (
                "{'k': {'a': 'b'}}",
                "options['k'] must be a str or a list of str, got dict",
            ),
            ("{'k': ['a', 1]}", "options['k'][1] must be a str, got int"),
            ("{'k': [['a']]}", "options['k'][0] must be a str, got list"),
            (
                "{'k': ('a', None)}",
                "options['k'][1] must be a str, got NoneType",
            ),
            ("{1: 'a'}", "options keys must be str, got int"),
            ("['a']", "options must be a dict"),
        ] {
            let err = extract_options(expr)
                .map(|_| ())
                .expect_err(&format!("{expr} must be rejected"));
            Python::attach(|py| {
                assert!(
                    err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{expr}: expected TypeError, got {err}"
                );
            });
            assert!(
                err.to_string().contains(needle),
                "{expr}: message {err} should contain {needle:?}"
            );
        }
    }
}
