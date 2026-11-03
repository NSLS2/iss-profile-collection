"""Exercise the Tiled startup wiring without beamline hardware or credentials."""

import ast
from pathlib import Path
from types import SimpleNamespace


STARTUP = Path(__file__).resolve().parents[1] / 'startup'


class Catalog:
    def __init__(self):
        self.path = []
        self.context = SimpleNamespace(http_client=SimpleNamespace(headers={}))

    def __getitem__(self, key):
        self.path.append(key)
        return self


def run_statements(path, statements, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if statements(node)]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def test_migration_clients_and_single_writer():
    catalogs = []
    subscriptions = []
    documents = []

    def from_profile(profile, **kwargs):
        assert profile == 'nsls2'
        catalog = Catalog()
        catalogs.append(catalog)
        return catalog

    def configure_base(user_ns, *args, **kwargs):
        assert args == ()
        assert kwargs['publish_documents_with_kafka'] == 'iss'

    class Writer:
        def __init__(self, *, client, **kwargs):
            assert client is catalogs[1]

        def __call__(self, name, doc):
            documents.append((name, doc))

    engine = SimpleNamespace(md={'data_session': 'pass-123'}, subscribe=subscriptions.append)
    namespace = {
        'from_profile': from_profile,
        'os': SimpleNamespace(environ={'TILED_BLUESKY_WRITING_API_KEY_ISS': 'test-key'}),
        'nslsii': SimpleNamespace(configure_base=configure_base),
        'get_ipython': lambda: SimpleNamespace(user_ns={}),
        'RE': engine,
        'TiledWriter': Writer,
        'BufferingWrapper': lambda writer: writer,
        'patch_descriptor': object(), 'patch_resource': object(), 'patch_datum': object(),
        'MIMETYPE_LOOKUP': {},
    }

    def startup_statement(node):
        if isinstance(node, ast.Assign):
            return any(isinstance(target, ast.Name) and target.id in {
                'tiled_reading_client', 'tiled_writing_client'
            } or isinstance(target, ast.Subscript) for target in node.targets)
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'configure_base')

    run_statements(STARTUP / '00-startup.py', startup_statement, namespace)
    assert [catalog.path for catalog in catalogs] == [['iss', 'migration'], ['iss', 'migration']]
    assert catalogs[1].context.http_client.headers['tiled-qos'] == 'acquisition'

    def writer_statement(node):
        return (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'tw'
                    or isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == 'RE' and target.value.attr == 'md'
                    for target in node.targets)
                or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'subscribe')

    run_statements(STARTUP / '02-tiled-writer.py', writer_statement, namespace)
    assert len(subscriptions) == 1
    assert engine.md['tiled_access_tags'] == ('pass-123',)
    for callback in subscriptions:
        callback('start', {'uid': 'test'})
    assert documents == [('start', {'uid': 'test'})]


def test_gui_client_targets_migration():
    tree = ast.parse((STARTUP / '99-gui.py').read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'xlive_tiled_client'
                              for target in node.targets))
    catalog = Catalog()
    namespace = {'from_uri': lambda *args, **kwargs: catalog}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '99-gui.py', 'exec'), namespace)
    assert catalog.path == ['iss', 'migration']
    assert namespace['xlive_tiled_client'] is catalog
