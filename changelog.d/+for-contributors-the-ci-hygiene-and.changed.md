For contributors: the CI, Hygiene and SonarQube Cloud workflows also run on
GitHub merge-queue groups, a prerequisite for enabling the queue on `main`.
Hygiene checks the queue's candidate commit against `main` with the public
rules; SonarQube Cloud skips its jobs there, since the quality gate already
ran on the pull request and the push to `main` reports on the merged commit.
