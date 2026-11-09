#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
r"""
Define the command line interface.

.. click:: formulas.cli:cli
   :prog: formulas
   :show-nested:

"""
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import re
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass

import click
import click_log
import schedula as sh

from .excel import ExcelModel
from .excel import _convert_complex, _file2books, _res2books
from .functions.date import xdate
from .ranges import Ranges

ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_WORKER_MODEL = ContextVar('worker_model')

log = logging.getLogger('formulas.cli')


class _Logger(logging.Logger):
    def setLevel(self, level):
        super(_Logger, self).setLevel(level)
        frmt = "%(asctime)-15s:%(levelname)5.5s:%(name)s:%(message)s"
        logging.basicConfig(level=level, format=frmt)
        rlog = logging.getLogger()
        # because `basicConfig()` does not reconfig root-logger when re-invoked.
        rlog.level = level
        logging.captureWarnings(True)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)


logger = _Logger('cli')
click_log.basic_config(logger)


class CliError(click.ClickException):
    pass


@dataclass
class RenderSpec:
    ref: str
    key: str | None = None


@dataclass
class Scenario:
    run_id: int
    run_name: str
    overwrites: dict
    compute_refs: list
    render_specs: list


@dataclass(frozen=True)
class BuildConfig:
    files: tuple
    output_refs: tuple
    output_file: str | None
    circular: bool


@dataclass(frozen=True)
class TestConfig:
    files: tuple
    against: tuple
    overwrites: dict
    output_refs: tuple
    circular: bool
    tolerance: float
    absolute_tolerance: float
    summary: bool


@dataclass(frozen=True)
class ServeConfig:
    files: tuple
    host: str
    port: int
    circular: bool


@click.group(
    help='Work with Excel and JSON spreadsheet models from the command line.',
    epilog="Use 'formulas COMMAND --help' for command-specific options and examples."
)
def cli():
    pass


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _slugify(value):
    value = re.sub(r'[^A-Za-z0-9._-]+', '-', value.strip())
    value = value.strip('-')
    return value or 'run'


def _parse_scalar_value(raw):
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return raw[1:-1]
    upper = raw.upper()
    if upper == 'TRUE':
        return True
    if upper == 'FALSE':
        return False
    if ISO_DATE_RE.match(raw):
        y, m, d = map(int, raw.split('-'))
        return int(xdate(y, m, d))
    try:
        if any(c in raw for c in '.eE'):
            return float(raw)
        return int(raw)
    except ValueError as exc:
        raise CliError(
            'Invalid scalar value `%s`. Use "..." for strings or YYYY-MM-DD '
            'for dates.' % raw
        ) from exc


def _parse_assignment(value):
    left, sep, right = value.partition('=')
    if not sep:
        raise CliError('Expected KEY=VALUE format: `%s`.' % value)
    return left.strip(), _parse_scalar_value(right.strip())


def _parse_outs_file(path):
    data = _load_json(path)
    if not isinstance(data, list):
        raise CliError('`--outs` JSON must be a list.')
    return [str(v) for v in data]


def _parse_render_entry(entry):
    if isinstance(entry, str):
        ref, sep, key = entry.partition('=')
        return RenderSpec(ref.strip(), key.strip() or None if sep else None)
    if isinstance(entry, dict) and 'ref' in entry:
        key = entry.get('key')
        return RenderSpec(str(entry['ref']).strip(), None if key is None else str(key))
    raise CliError('Invalid render specification `%r`.' % (entry,))


def _parse_renders_file(path):
    data = _load_json(path)
    if not isinstance(data, list):
        raise CliError('`--renders` JSON must be a list.')
    return [_parse_render_entry(entry) for entry in data]


def _normalize_matrix(value):
    if isinstance(value, list):
        if value and not isinstance(value[0], list):
            return [[v] for v in value]
        return value
    return value


def _build_input_payload(overwrites):
    payload = {}
    for ref, value in overwrites.items():
        if ':' in ref:
            matrix = _normalize_matrix(value)
            payload[ref] = Ranges().push(ref, value=matrix)  # type: ignore[arg-type]
        else:
            payload[ref] = value
    return payload


def _load_model(files, circular=False):
    model = ExcelModel()
    quiet_tqdm = ExcelModel.complete.__globals__['tqdm'].tqdm

    class QuietTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *args, **kwargs):
            return None

        def refresh(self):
            return None

        total = 0

    ExcelModel.complete.__globals__['tqdm'].tqdm = QuietTqdm
    try:
        for path in files:
            ext = osp.splitext(path)[1].lower()
            if ext == '.json':
                model.from_dict(_load_json(path), assemble=False)
            elif ext in ('.xlsx', '.ods'):
                model.loads(path)
            else:
                raise CliError('Unsupported input file `%s`.' % path)
        model.finish(complete=True, circular=circular)
        return model
    finally:
        ExcelModel.complete.__globals__['tqdm'].tqdm = quiet_tqdm


def _extract_value(value):
    if isinstance(value, Ranges):
        data = value.value.tolist()  # type: ignore[union-attr]
        if len(data) == 1 and len(data[0]) == 1:
            return data[0][0]
        return data
    return value


def _render_output(solution, render_specs, computed_refs):
    if not render_specs:
        refs = computed_refs or [k for k in solution if not isinstance(k, sh.Token)]
        data = {ref: _extract_value(solution[ref]) for ref in refs if ref in solution}
        return data
    rendered, used = {}, {}
    for spec in render_specs:
        if spec.ref not in solution:
            continue
        key = spec.key or spec.ref
        if key in used and used[key] != spec.ref:
            raise CliError('Duplicate render key `%s`.' % key)
        used[key] = spec.ref
        rendered[key] = _extract_value(solution[spec.ref])
    return rendered


def _write_json_result(data, output_file=None):
    text = json.dumps(data, indent=2, sort_keys=True)
    if output_file:
        with open(output_file, 'w') as fh:
            fh.write(text)
            fh.write('\n')
    else:
        click.echo(text)


def _normalize_out_refs(outs_inline, outs_file):
    compute_refs = list(outs_inline)
    if outs_file:
        compute_refs.extend(_parse_outs_file(outs_file))
    return list(dict.fromkeys(compute_refs))


def _write_batch_json(output_dir, run_id, run_name, data):
    os.makedirs(output_dir, exist_ok=True)
    name = _slugify(run_name or 'run')
    fpath = osp.join(output_dir, 'run-%04d-%s.json' % (run_id, name))
    _write_json_result(data, output_file=fpath)
    return fpath


def _write_excel_result(model, solution, output_dir, run_id, run_name):
    name = _slugify(run_name or 'run')
    run_dir = osp.join(output_dir, 'run-%04d-%s' % (run_id, name))
    books = deepcopy(model.books)
    model.write(books=books, solution=solution, dirpath=run_dir)
    saved = [osp.join(run_dir, book_name) for book_name in books]
    return run_dir, saved


def _collect_missing_messages(model, overwrites, compute_refs, render_specs):
    nodes = set(model.dsp.data_nodes)
    warnings = []
    for ref in overwrites:
        if ref not in nodes:
            warnings.append('Missing overwrite reference: %s' % ref)
    for ref in compute_refs:
        if ref not in nodes:
            warnings.append('Missing output reference: %s' % ref)
    for spec in render_specs:
        if spec.ref not in nodes:
            warnings.append('Missing render reference: %s' % spec.ref)
    return warnings


def _read_batch(path):
    data = _load_json(path)
    if not isinstance(data, list):
        raise CliError('`--batch` JSON must be a list.')
    return data


def _normalize_overwrites(overwrites):
    overwrite_map = {}
    for item in overwrites:
        ref, value = _parse_assignment(item)
        if ':' in ref:
            raise CliError('Ranges can be overwritten only using `--batch`.')
        overwrite_map[ref] = value
    return overwrite_map


def _normalize_top_level(outs_inline, outs_file, renders_inline, renders_file):
    compute_refs = _normalize_out_refs(outs_inline, outs_file)
    render_specs = [_parse_render_entry(item) for item in renders_inline]
    if renders_file:
        render_specs.extend(_parse_renders_file(renders_file))
    compute_refs = list(dict.fromkeys(compute_refs + [spec.ref for spec in render_specs]))
    return compute_refs, render_specs


def _normalize_scenarios(batch_data, overwrite_map, compute_refs, render_specs):
    if batch_data is None:
        batch_data = [{'name': 'default'}]
    scenarios = []
    for idx, run in enumerate(batch_data, start=1):
        run_name = str(run.get('name') or 'run')
        run_overwrites = dict(overwrite_map)
        batch_overwrites = run.get('overwrite', {})
        if not isinstance(batch_overwrites, dict):
            raise CliError('Batch `overwrite` must be an object.')
        run_overwrites.update(batch_overwrites)

        run_compute_refs = list(compute_refs)
        run_render_specs = list(render_specs)
        if 'outs' in run:
            entries = run['outs']
            if not isinstance(entries, list):
                raise CliError('Batch `outs` must be a list.')
            run_compute_refs = [str(v) for v in entries]
        if 'renders' in run:
            entries = run['renders']
            if not isinstance(entries, list):
                raise CliError('Batch `renders` must be a list.')
            run_render_specs = [_parse_render_entry(v) for v in entries]
        run_compute_refs = list(dict.fromkeys(
            run_compute_refs + [spec.ref for spec in run_render_specs]
        ))
        scenarios.append(Scenario(
            run_id=idx,
            run_name=run_name,
            overwrites=run_overwrites,
            compute_refs=run_compute_refs,
            render_specs=run_render_specs,
        ))
    return scenarios


def _execute_scenario(model, scenario, output_format, output_dir):
    warnings = _collect_missing_messages(
        model, scenario.overwrites, scenario.compute_refs, scenario.render_specs
    )
    solution = model.calculate(
        inputs=_build_input_payload(scenario.overwrites),
        outputs=scenario.compute_refs or None
    )
    rendered = _render_output(solution, scenario.render_specs, scenario.compute_refs)
    if scenario.render_specs and not rendered:
        raise CliError('Run `%s` produced no renderable outputs.' % scenario.run_name)

    result = {
        'run_id': scenario.run_id,
        'name': scenario.run_name,
        'warnings': warnings,
        'status': 'ok',
    }
    if output_format == 'json':
        if output_dir:
            path = _write_batch_json(output_dir, scenario.run_id, scenario.run_name, rendered)
            result['path'] = path
        else:
            result['data'] = rendered
    else:
        run_dir, saved = _write_excel_result(
            model, solution, output_dir, scenario.run_id, scenario.run_name
        )
        if not saved:
            raise CliError('Run `%s` produced no output files.' % scenario.run_name)
        result['dir'] = run_dir
        result['files'] = saved
    return result


def _init_worker(files, circular):
    _WORKER_MODEL.set(_load_model(files, circular=circular))


def _run_worker(payload):
    model = _WORKER_MODEL.get()
    scenario = Scenario(
        run_id=payload['run_id'],
        run_name=payload['run_name'],
        overwrites=payload['overwrites'],
        compute_refs=payload['compute_refs'],
        render_specs=[RenderSpec(**item) for item in payload['render_specs']],
    )
    try:
        return _execute_scenario(
            model, scenario, payload['output_format'], payload['output_dir']
        )
    except Exception as exc:  # pragma: no cover - exercised by CLI tests via result payloads.
        return {
            'run_id': scenario.run_id,
            'name': scenario.run_name,
            'warnings': _collect_missing_messages(
                model, scenario.overwrites,
                scenario.compute_refs, scenario.render_specs
            ),
            'status': 'error',
            'error_type': type(exc).__name__,
            'error': str(exc),
        }


def _execute_batch(files, circular, scenarios, output_format, output_dir, processes):
    payloads = [{
        'run_id': scenario.run_id,
        'run_name': scenario.run_name,
        'overwrites': scenario.overwrites,
        'compute_refs': scenario.compute_refs,
        'render_specs': [asdict(spec) for spec in scenario.render_specs],
        'output_format': output_format,
        'output_dir': output_dir,
    } for scenario in scenarios]
    if processes == 1:
        _init_worker(files, circular)
        results = [_run_worker(payload) for payload in payloads]
        return results

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=processes, initializer=_init_worker,
                  initargs=(files, circular)) as pool:
        return pool.map(_run_worker, payloads)


def _emit_warnings(results):
    for result in results:
        for warning in result.get('warnings', ()):
            log.warning(warning)


def _normalize_build_config(files, outs_inline, outs_file, output_file, circular):
    if not files:
        raise CliError('At least one input file is required.')
    return BuildConfig(
        files=tuple(files),
        output_refs=tuple(_normalize_out_refs(outs_inline, outs_file)),
        output_file=output_file,
        circular=circular,
    )


def _collect_missing_output_messages(model, refs):
    return [
        'Missing output reference: %s' % ref
        for ref in refs if ref not in model.dsp.data_nodes
    ]


def _resolve_build_refs(model, refs):
    if not refs:
        return [], []
    valid = [ref for ref in refs if ref in model.dsp.data_nodes]
    missing = _collect_missing_output_messages(model, refs)
    if not valid:
        raise CliError('All requested output references are missing.')
    return valid, missing


def _collect_dependency_refs(model, output_refs):
    exported = model.to_dict()
    if not output_refs:
        return set(exported)
    dsp = model.dsp.shrink_dsp(outputs=list(output_refs))
    nodes = {node for node in dsp.data_nodes if not isinstance(node, sh.Token)}
    nodes.update(output_refs)
    return nodes.intersection(exported)


def _export_model_dict(model, output_refs):
    exported = model.to_dict()
    if not output_refs:
        return exported
    valid_nodes = _collect_dependency_refs(model, output_refs)
    return {key: value for key, value in exported.items() if key in valid_nodes}


def _run_build(config):
    model = _load_model(config.files, circular=config.circular)
    output_refs, warnings = _resolve_build_refs(model, config.output_refs)
    for warning in warnings:
        log.warning(warning)
    exported = _export_model_dict(model, output_refs)
    if not exported:
        raise CliError('Build produced no exportable model entries.')
    _write_json_result(exported, output_file=config.output_file)


def _default_against(files):
    targets = [
        path for path in files
        if osp.splitext(path)[1].lower() in ('.xlsx', '.ods')
    ]
    if not targets:
        raise CliError('`--against` is required when no workbook inputs are provided.')
    return tuple(targets)


def _normalize_test_config(files, against, overwrites, outs_inline, outs_file,
                           tolerance, absolute_tolerance, circular, summary):
    if not files:
        raise CliError('At least one input file is required.')
    return TestConfig(
        files=tuple(files),
        against=tuple(against) if against else _default_against(files),
        overwrites=_normalize_overwrites(overwrites),
        output_refs=tuple(_normalize_out_refs(outs_inline, outs_file)),
        circular=circular,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        summary=summary,
    )


def _format_summary_table(result):
    rows = [
        ('status', 'PASS' if result['passed'] else 'FAIL'),
        ('targets', str(result['targets'])),
        ('scoped', 'yes' if result['scoped'] else 'no'),
        ('outputs', str(result['outputs'])),
        ('missing_outputs', str(result['missing_outputs'])),
        ('missing_overwrites', str(result['missing_overwrites'])),
    ]
    width = max(len(key) for key, _ in rows)
    return '\n'.join('%-*s  %s' % (width, key, value) for key, value in rows)


def _column_number_to_name(index):
    chars = []
    while index:
        index, remainder = divmod(index - 1, 26)
        chars.append(chr(ord('A') + remainder))
    return ''.join(reversed(chars))


def _book_range_refs(rng):
    c1 = rng.get('c1') or 'A'
    c2 = rng.get('c2') or c1
    r1 = int(rng.get('r1') or 1)
    r2 = int(rng.get('r2') or r1)

    def col_to_num(col):
        value = 0
        for char in col.upper():
            value = value * 26 + ord(char) - ord('A') + 1
        return value

    refs = []
    for row in range(r1, r2 + 1):
        for col in range(col_to_num(c1), col_to_num(c2) + 1):
            refs.append('%s%d' % (_column_number_to_name(col), row))
    return refs


def _solution_refs_to_book_map(solution):
    refs = {}
    for key, value in solution.items():
        if isinstance(key, sh.Token):
            continue
        if isinstance(value, Ranges):
            ranges = value.ranges
        else:
            try:
                ranges = Ranges().push(key, value).ranges
            except ValueError:
                continue
        for rng in ranges:
            filename = rng.get('filename', '')
            sheet = rng.get('sheet', '')
            if not filename or not sheet:
                continue
            book_key = filename.upper()
            sheet_key = sheet.upper()
            refs.setdefault(book_key, {}).setdefault(sheet_key, set()).update(
                _book_range_refs(rng)
            )
    return refs


def _filter_book_data(books_data, allowed_refs):
    filtered = {}
    for book, sheets in books_data.items():
        book_allowed = allowed_refs.get(book, {})
        if not book_allowed:
            continue
        filtered_sheets = {}
        for sheet, cells in sheets.items():
            allowed_cells = book_allowed.get(sheet, set())
            subset = {cell: value for cell, value in cells.items() if cell in allowed_cells}
            if subset:
                filtered_sheets[sheet] = subset
        if filtered_sheets:
            filtered[book] = filtered_sheets
    return filtered


def _run_test(config):
    model = _load_model(config.files, circular=config.circular)
    missing_overwrite_messages = _collect_missing_messages(
        model, config.overwrites, (), ()
    )
    for warning in missing_overwrite_messages:
        log.warning(warning)

    output_refs, missing_output_messages = _resolve_build_refs(model, config.output_refs)
    for warning in missing_output_messages:
        log.warning(warning)

    books = model.books
    solution = model.calculate(
        inputs=_build_input_payload(config.overwrites),
        outputs=output_refs or None
    )
    if output_refs:
        actual = _convert_complex(_res2books(model.write(
            books=deepcopy(model.books), solution=solution
        )))
        target = _convert_complex(_file2books(*config.against))
        allowed_refs = _solution_refs_to_book_map(solution)
        report = model.compare(
            target=_filter_book_data(target, allowed_refs),
            actual=_filter_book_data(actual, allowed_refs),
            tolerance=config.tolerance,
            absolute_tolerance=config.absolute_tolerance,
        )
    else:
        report = model.compare(
            *config.against,
            solution=solution,
            books=books,
            tolerance=config.tolerance,
            absolute_tolerance=config.absolute_tolerance,
        )
    return {
        'passed': report == 'No differences.',
        'report': report,
        'summary': config.summary,
        'targets': len(config.against),
        'scoped': bool(output_refs),
        'outputs': len(output_refs) if output_refs else 'all',
        'missing_outputs': len(missing_output_messages),
        'missing_overwrites': len(missing_overwrite_messages),
    }


def _emit_test_result(result):
    if result['summary']:
        click.echo(_format_summary_table(result))
        click.echo()
    click.echo(result['report'])
    if not result['passed']:
        raise click.exceptions.Exit(1)


def _normalize_serve_config(files, host, port, circular):
    if not files:
        raise CliError('At least one input file is required.')
    return ServeConfig(
        files=tuple(files),
        host=host,
        port=port,
        circular=circular
    )


def _set_serve_environment(config):
    os.environ['FORMULAS_SERVE_FILES'] = json.dumps(config.files)
    os.environ['FORMULAS_SERVE_CIRCULAR'] = json.dumps(config.circular)


@cli.command(
    help='Export a reusable JSON model from workbook and JSON inputs.'
)
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False), metavar='FILES...')
@click.option('--out', 'outs_inline', multiple=True,
              help='Cell or range to keep in the exported model, together with its dependencies. Repeatable.')
@click.option('--outs', 'outs_file', type=click.Path(exists=True, dir_okay=False),
              help='JSON file with a list of cells or ranges to keep in the exported model.')
@click.option('--output-file', type=click.Path(dir_okay=False),
              help='Write the exported JSON model to a file instead of stdout.')
@click.option('--circular/--no-circular', default=False,
              help='Enable or disable circular reference solving while loading the model.')
def build(files, outs_inline, outs_file, output_file, circular):
    logging.basicConfig(level=logging.WARNING)
    config = _normalize_build_config(files, outs_inline, outs_file, output_file, circular)
    _run_build(config)


@cli.command(
    name='test',
    help='Test whether formulas reproduces the reference workbook values.'
)
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False), metavar='FILES...')
@click.option('--against', 'against', multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Reference workbook to compare against. Repeatable. Defaults to workbook inputs from FILES when omitted.')
@click.option('--overwrite', 'overwrites', multiple=True,
              help='Override a scalar cell with CELL=VALUE before testing. Strings must use double quotes; dates must use YYYY-MM-DD.')
@click.option('--out', 'outs_inline', multiple=True,
              help='Cell or range to test. Limits the comparison to the selected outputs. Repeatable.')
@click.option('--outs', 'outs_file', type=click.Path(exists=True, dir_okay=False),
              help='JSON file with a list of cells or ranges to test. Limits the comparison to the selected outputs.')
@click.option('--tolerance', type=float, default=0.0, show_default=True,
              help='Relative tolerance passed to the comparison engine.')
@click.option('--absolute-tolerance', type=float, default=.000001, show_default=True,
              help='Absolute tolerance passed to the comparison engine.')
@click.option('--summary', is_flag=True,
              help='Print a small text summary table before the comparison report.')
@click.option('--circular/--no-circular', default=False,
              help='Enable or disable circular reference solving while loading the model.')
@click_log.simple_verbosity_option(logger)
def test_command(files, against, overwrites, outs_inline, outs_file, tolerance,
                 absolute_tolerance, summary, circular):
    config = _normalize_test_config(
        files, against, overwrites, outs_inline, outs_file,
        tolerance, absolute_tolerance, circular, summary
    )
    _emit_test_result(_run_test(config))


@cli.command(
    help='Start a Flask API server for a loaded spreadsheet model.'
)
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False), metavar='FILES...')
@click.option('--host', default='127.0.0.1', show_default=True,
              help='Host interface to bind the API server to.')
@click.option('--port', default=5000, type=int, show_default=True,
              help='Port to bind the API server to.')
@click.option('--circular/--no-circular', default=False,
              help='Enable or disable circular reference solving while loading the model.')
@click_log.simple_verbosity_option(logger)
@click.pass_context
def serve(ctx, files, host, port, circular):
    config = _normalize_serve_config(files, host, port, circular)
    _set_serve_environment(config)
    from schedula.utils.form import cli as _cli
    ctx.params.clear()
    ctx.params['folder'] = osp.dirname(osp.dirname(__file__))
    ctx.params['app'] = 'formulas.app:create_app()'
    cmd = _cli.run
    ctx.args.extend(['--bind', f'{host}:{port}'])
    return cmd.invoke(ctx)


@cli.command(
    help='Calculate spreadsheet models from workbook and JSON inputs.'
)
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False), metavar='FILES...')
@click.option('--overwrite', 'overwrites', multiple=True,
              help='Override a scalar cell with CELL=VALUE. Strings must use double quotes; dates must use YYYY-MM-DD.')
@click.option('--batch', type=click.Path(exists=True, dir_okay=False),
              help='JSON file with a list of run definitions. Required for range overwrites and batch execution.')
@click.option('--out', 'outs_inline', multiple=True,
              help='Output cell or range to compute. Reduces computation scope. Repeatable.')
@click.option('--outs', 'outs_file', type=click.Path(exists=True, dir_okay=False),
              help='JSON file with a list of cells or ranges to compute. Reduces computation scope.')
@click.option('--render', 'renders_inline', multiple=True,
              help='Cell or range to emit in the result payload, optionally as REF=KEY. Also adds the ref to the computation scope. Repeatable.')
@click.option('--renders', 'renders_file', type=click.Path(exists=True, dir_okay=False),
              help='JSON file with a list of render specs. Reduces emitted output and implicitly extends computation scope.')
@click.option('--output-format', type=click.Choice(['excel', 'json']), required=True,
              help='Write results as JSON or Excel artifacts.')
@click.option('--output-dir', type=click.Path(file_okay=False),
              help='Output directory for batch JSON results or Excel run folders.')
@click.option('--output-file', type=click.Path(dir_okay=False),
              help='Output file for non-batch JSON results.')
@click.option('--processes', type=int, default=1, show_default=True,
              help='Number of worker processes for batch execution. Only valid with --batch.')
@click.option('--circular/--no-circular', default=False,
              help='Enable or disable circular reference solving while loading the model.')
@click_log.simple_verbosity_option(logger)
def calc(files, overwrites, batch, outs_inline, outs_file, renders_inline,
         renders_file, output_format, output_dir, output_file, processes, circular):
    if not files:
        raise CliError('At least one input file is required.')
    if processes < 1:
        raise CliError('`--processes` must be greater than 0.')
    if processes > 1 and not batch:
        raise CliError('`--processes` can be used only with `--batch`.')
    if output_format == 'excel' and not output_dir:
        raise CliError('`--output-dir` is required for excel output.')
    if batch and output_file:
        raise CliError('`--output-file` cannot be used with `--batch`.')
    if output_format == 'json' and batch and not output_dir:
        raise CliError('`--output-dir` is required for batch json output.')

    overwrite_map = _normalize_overwrites(overwrites)
    compute_refs, render_specs = _normalize_top_level(
        outs_inline, outs_file, renders_inline, renders_file
    )

    if batch:
        batch_data = _read_batch(batch)
        scenarios = _normalize_scenarios(batch_data, overwrite_map, compute_refs, render_specs)
        results = _execute_batch(files, circular, scenarios, output_format, output_dir, processes)
        results = sorted(results, key=lambda item: item['run_id'])
        _emit_warnings(results)
        click.echo(json.dumps(results, indent=2, sort_keys=True))
        if any(item['status'] != 'ok' for item in results):
            raise click.exceptions.Exit(1)
        return

    model = _load_model(files, circular=circular)
    for warning in _collect_missing_messages(model, overwrite_map, compute_refs, render_specs):
        log.warning(warning)

    scenario = Scenario(
        run_id=1,
        run_name='default',
        overwrites=overwrite_map,
        compute_refs=compute_refs,
        render_specs=render_specs,
    )
    result = _execute_scenario(model, scenario, output_format, output_dir=None)
    if output_format == 'json':
        if output_file:
            _write_json_result(result['data'], output_file=output_file)
        else:
            _write_json_result(result['data'])
        return
    raise CliError('Non-batch excel output requires `--output-dir`.')
