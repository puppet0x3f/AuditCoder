# Anonymous publication

The release contains source files and benchmark inputs only. It excludes the
original Git history, private configuration, cookies, local databases, editor
files, manuscript/PDF assets, historical result directories, and API traces.
Absolute historical trace paths in corpus metadata are cleared. All runtime
defaults resolve inside this repository, and no runner loads a parent
workspace's `.env`.

Third-party baseline copyright holders, dataset attribution, and upstream
project links remain in their notices. They credit dependencies; they do not
identify this release's authors. The AuditCoder license uses the existing
generic contributor name. No author citation block is included.

## File-content check

```bash
python scripts/check_anonymity.py
```

The check reports file names and line numbers without printing matched secret
values. It checks distributable text for common API-token formats, private-key
headers, explicit author metadata, and local user paths. Local `.env`, ignored
run outputs, environments, and Git internals are outside its scope. It cannot
establish anonymity of a GitHub account, Git history, prose, or unrecognized
credential formats.

## Create a fresh anonymous repository

Use a separate anonymous GitHub account/repository for the review submission.
Do not publish this folder under an identifying personal or institutional
account. Create fresh commits inside this folder; copying an existing `.git`
directory would restore its history and identities.

The commands below create a local anonymous initial commit. The placeholder
email is intentionally non-deliverable. They do not publish anything:

```bash
git init -b main
git config --local user.name "Anonymous Authors"
git config --local user.email "anonymous@example.invalid"
git config --local commit.gpgsign false
git config --local tag.gpgsign false
git config --local core.hooksPath /dev/null
git add .
git diff --cached --stat
git diff --cached --check
git -c user.name="Anonymous Authors" -c user.email="anonymous@example.invalid" \
  commit -m "Anonymous code release"
git log -1 --format=fuller
```

Verify both author and committer identities in the last command. If your shell
sets `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, clear those overrides before committing.
Configure the remote for the anonymous account only after reviewing the staged
files. Avoid identifying issue links, release attachments, collaborators,
organization names, or external project pages in the review repository.

Model runs write local paths and diagnostic text into ignored `results/` files.
Review and sanitize those files separately if they are later added to a public
supplement. `.gitignore` does not protect a file added with `git add -f`.
