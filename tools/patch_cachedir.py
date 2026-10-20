import io

p = "pipeline/project.py"
s = io.open(p, encoding="utf-8").read()

old = '''class Project:
    """One film. Owns its paths and its artifact store."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root)
        self.store = Store(self.path(".cache"))
'''
new = '''# Where the artifact store lives, when it is not inside the project.
#
# The store is the only part of a project that grows without bound -- a minute
# of film is a thousand PNGs, and §4.2 has "the store never garbage-collects"
# on its list. On a machine whose system drive is nearly full, or whose project
# directory is inside a cloud-synced folder that will dutifully upload every
# rendered frame, the store needs to be somewhere else. Everything else in a
# project is small, diffable and worth keeping beside the code.
#
# Resolution order, first hit wins:
#   1. $PIPELINE_CACHE_DIR/<project name>   -- one setting for every project
#   2. project.json's "cache_dir"           -- per project; relative paths are
#                                              resolved against the project root
#   3. <project>/.cache                     -- unchanged default
CACHE_DIR_ENV = "PIPELINE_CACHE_DIR"


class Project:
    """One film. Owns its paths and its artifact store."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root)
        self.cache_root = self._resolve_cache_root()
        self.store = Store(self.cache_root)

    def _resolve_cache_root(self):
        shared = os.environ.get(CACHE_DIR_ENV)
        if shared:
            return os.path.abspath(os.path.join(shared, self.name))
        setting = jsonio.read(self.path("project.json"), default={}) \\
            .get("cache_dir")
        if setting:
            return os.path.abspath(os.path.join(self.root, setting))
        return self.path(".cache")

    def cache(self, *parts):
        """A path inside the artifact store, wherever the store lives.

        Every ``project.path(".cache", ...)`` in the codebase became a call to
        this, because a store that moves has to move whole: leaving the work
        directory or the run summary behind on the full drive would defeat the
        point.
        """
        joined = "/".join(str(part) for part in parts if part not in (None, ""))
        return os.path.join(self.cache_root, *joined.split("/")) if joined \\
            else self.cache_root
'''
assert old in s, "Project.__init__ not found"
s = s.replace(old, new)

# project.json gains the setting, so `pipeline init` writes a project that says
# where its store is even when that is the default.
old = '''    "aligner_backend": "auto",
    "silence_lead_policy": "trim",
}'''
new = '''    "aligner_backend": "auto",
    "silence_lead_policy": "trim",
    # null means <project>/.cache. See CACHE_DIR_ENV above.
    "cache_dir": None,
}'''
assert old in s, "PROJECT_DEFAULTS not found"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("project.py patched")

for path, subs in (
    ("pipeline/cli.py", [
        ('project.path(".cache", "validate.json")',
         'project.cache("validate.json")')]),
    ("pipeline/graph.py", [
        ('dir=_ensure(self.project.path(".cache", "work"))',
         'dir=_ensure(self.project.cache("work"))'),
        ('path = self.project.path(".cache", "last_run.json")',
         'path = self.project.cache("last_run.json")')]),
    ("pipeline/stages/_selftest.py", [
        ('jsonio.read(project.path(".cache", *PLAN_FILE), default=None)',
         'jsonio.read(project.cache(*PLAN_FILE), default=None)')]),
):
    text = io.open(path, encoding="utf-8").read()
    for old_sub, new_sub in subs:
        assert old_sub in text, (path, old_sub)
        text = text.replace(old_sub, new_sub)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(path, "patched")
