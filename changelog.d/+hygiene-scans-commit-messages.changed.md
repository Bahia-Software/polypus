For contributors: the Hygiene check also scans the messages of the commits
it analyses (merge base to head on pull requests, base to head on merge-queue
groups and pushes to `main`) with the same public rules as file content, plus
the private patterns on `main`, and flags internal `bahia<N>` host names
alongside login-node names. Commit messages admit no allow-list exceptions.
