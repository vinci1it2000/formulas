#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import io
import json
import logging
import os
import os.path as osp
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    from click.testing import CliRunner
    from formulas.app import create_app
    from formulas.cli import cli, _normalize_serve_config, _set_serve_environment
except ImportError:
    pass
EXTRAS = os.environ.get('EXTRAS', 'all')
mydir = osp.join(osp.dirname(__file__), 'test_files')


@unittest.skipIf(EXTRAS not in ('all', 'cli'), 'Not for extra %s.' % EXTRAS)
class TestCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp(prefix='formulas-cli-')
        self.excel = osp.join(mydir, 'excel.xlsx')
        self.json_model = osp.join(self.tmpdir, 'model.json')
        with open(self.json_model, 'w') as fh:
            json.dump({'A1': 5, 'B1': '=A1*2'}, fh)

    def write_json(self, name, data):
        path = osp.join(self.tmpdir, name)
        with open(path, 'w') as fh:
            json.dump(data, fh)
        return path

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_calc_json_output_file(self):
        output_file = osp.join(self.tmpdir, 'result.json')
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--out', "'[excel.xlsx]DATA'!C2",
            '--render', "'[excel.xlsx]DATA'!C2=result",
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
            '--overwrite', "'[excel.xlsx]DATA'!B3=1",
            '--output-format', 'json',
            '--output-file', output_file,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        with open(output_file) as fh:
            data = json.load(fh)
        self.assertEqual({'result': 10.0}, data)

    def test_calc_overwrite_string_value(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1="hello"',
            '--render', 'A1',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'A1': 'hello'}, json.loads(result.output))

    def test_calc_overwrite_boolean_value(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1=TRUE',
            '--render', 'A1',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'A1': True}, json.loads(result.output))

    def test_calc_overwrite_float_value(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1=1.5',
            '--render', 'A1',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'A1': 1.5}, json.loads(result.output))

    def test_calc_overwrite_iso_date_value(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1=2025-01-01',
            '--render', 'A1',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'A1': 45658}, json.loads(result.output))

    def test_calc_rejects_invalid_assignment(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1',
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Expected KEY=VALUE format', result.output)

    def test_calc_overwrite_false_value(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--overwrite', 'A1=FALSE',
            '--render', 'A1',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'A1': False}, json.loads(result.output))

    def test_calc_json_stdout_with_render_only(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--render', "'[excel.xlsx]DATA'!C2=result",
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
            '--overwrite', "'[excel.xlsx]DATA'!B3=1",
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'result': 10.0}, json.loads(result.output))

    def test_calc_mixed_inputs(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel, self.json_model,
            '--render', 'B1=json_result',
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'json_result': 10}, json.loads(result.output))

    def test_calc_outs_file(self):
        outs_file = self.write_json('outs.json', ['B1'])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--outs', outs_file,
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'B1': 10}, json.loads(result.output))

    def test_calc_renders_file_strings(self):
        renders_file = self.write_json('renders.json', ['B1=result'])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--renders', renders_file,
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'result': 10}, json.loads(result.output))

    def test_calc_renders_file_objects(self):
        renders_file = self.write_json('renders_objects.json', [{'ref': 'B1', 'key': 'result'}])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--renders', renders_file,
            '--output-format', 'json',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual({'result': 10}, json.loads(result.output))

    def test_calc_rejects_invalid_outs_json(self):
        outs_file = self.write_json('outs_invalid.json', {'ref': 'B1'})
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--outs', outs_file,
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--outs` JSON must be a list.', result.output)

    def test_calc_rejects_invalid_renders_json(self):
        renders_file = self.write_json('renders_invalid.json', {'ref': 'B1'})
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--renders', renders_file,
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--renders` JSON must be a list.', result.output)

    def test_calc_rejects_invalid_render_entry(self):
        renders_file = self.write_json('renders_bad_entry.json', [{'key': 'missing-ref'}])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--renders', renders_file,
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Invalid render specification', result.output)

    def test_calc_json_batch(self):
        batch_file = osp.join(self.tmpdir, 'batch.json')
        with open(batch_file, 'w') as fh:
            json.dump([
                {
                    'name': 'base',
                    'overwrite': {
                        "'[excel.xlsx]'!INPUT_A": 3,
                        "'[excel.xlsx]DATA'!B3": 1
                    },
                    'renders': ["'[excel.xlsx]DATA'!C2=result"]
                }
            ], fh)
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        files = [osp.basename(item['path']) for item in summary]
        self.assertEqual(1, len(files))
        with open(osp.join(self.tmpdir, files[0])) as fh:
            data = json.load(fh)
        self.assertEqual({'result': 10.0}, data)

    def test_calc_json_batch_parallel(self):
        batch_file = osp.join(self.tmpdir, 'batch_parallel.json')
        with open(batch_file, 'w') as fh:
            json.dump([
                {
                    'name': 'second',
                    'overwrite': {
                        "'[excel.xlsx]'!INPUT_A": 4,
                        "'[excel.xlsx]DATA'!B3": 1
                    },
                    'renders': ["'[excel.xlsx]DATA'!C2=result"]
                },
                {
                    'name': 'first',
                    'overwrite': {
                        "'[excel.xlsx]'!INPUT_A": 3,
                        "'[excel.xlsx]DATA'!B3": 1
                    },
                    'renders': ["'[excel.xlsx]DATA'!C2=result"]
                }
            ], fh)
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--processes', '2',
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        self.assertEqual([1, 2], [item['run_id'] for item in summary])
        self.assertEqual(['second', 'first'], [item['name'] for item in summary])

    def test_calc_batch_range_overwrite(self):
        batch_file = self.write_json('batch_range.json', [
            {
                'name': 'range',
                'overwrite': {'A1:A3': [2, 4, 6]},
                'renders': ['A1:A3=values']
            }
        ])
        model = self.write_json('range_model.json', {
            'A1:A3': 0
        })
        result = self.runner.invoke(cli, [
            'calc', model,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        with open(summary[0]['path']) as fh:
            data = json.load(fh)
        self.assertEqual({'values': [[2], [4], [6]]}, data)

    def test_calc_batch_range_overwrite_matrix(self):
        batch_file = self.write_json('batch_range_matrix.json', [
            {
                'name': 'matrix',
                'overwrite': {'A1:B2': [[1, 2], [3, 4]]},
                'renders': ['A1:B2=values']
            }
        ])
        model = self.write_json('range_matrix_model.json', {
            'A1:B2': 0
        })
        result = self.runner.invoke(cli, [
            'calc', model,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        with open(summary[0]['path']) as fh:
            data = json.load(fh)
        self.assertEqual({'values': [[1, 2], [3, 4]]}, data)

    def test_calc_excel_batch_output(self):
        batch_file = osp.join(self.tmpdir, 'batch_excel.json')
        with open(batch_file, 'w') as fh:
            json.dump([
                {
                    'name': 'excel',
                    'overwrite': {
                        "'[excel.xlsx]'!INPUT_A": 3,
                        "'[excel.xlsx]DATA'!B3": 1
                    }
                }
            ], fh)
        output_dir = osp.join(self.tmpdir, 'excel-runs')
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--output-format', 'excel',
            '--output-dir', output_dir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        self.assertEqual('ok', summary[0]['status'])
        self.assertTrue(osp.isdir(summary[0]['dir']))
        self.assertTrue(any(path.endswith('.XLSX') for path in summary[0]['files']))

    def test_calc_batch_repeated_blank_names(self):
        batch_file = self.write_json('batch_blank_names.json', [
            {'name': '   ', 'renders': ['B1=result']},
            {'name': '   ', 'renders': ['B1=result']}
        ])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.output)
        self.assertTrue(summary[0]['path'].endswith('run-0001-run.json'))
        self.assertTrue(summary[1]['path'].endswith('run-0002-run.json'))

    def test_calc_rejects_processes_without_batch(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--output-format', 'json',
            '--processes', '2',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--processes` can be used only with `--batch`.', result.output)

    def test_calc_rejects_non_positive_processes(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', self.write_json('batch_empty.json', []),
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
            '--processes', '0',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--processes` must be greater than 0.', result.output)

    def test_calc_rejects_no_files(self):
        result = self.runner.invoke(cli, [
            'calc',
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('At least one input file is required.', result.output)

    def test_calc_rejects_excel_without_output_dir(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--output-format', 'excel',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--output-dir` is required for excel output.', result.output)

    def test_calc_rejects_batch_json_without_output_dir(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', self.write_json('batch_no_dir.json', []),
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--output-dir` is required for batch json output.', result.output)

    def test_calc_rejects_output_file_with_batch(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', self.write_json('batch_output_file.json', []),
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
            '--output-file', osp.join(self.tmpdir, 'out.json'),
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--output-file` cannot be used with `--batch`.', result.output)

    def test_calc_rejects_non_batch_excel_after_model_execution(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--output-format', 'excel',
            '--output-dir', self.tmpdir,
            '--batch', self.write_json('empty_batch_for_excel.json', []),
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([], json.loads(result.output))

    def test_calc_rejects_unsupported_extension(self):
        bad_file = osp.join(self.tmpdir, 'unsupported.txt')
        with open(bad_file, 'w') as fh:
            fh.write('bad')
        result = self.runner.invoke(cli, [
            'calc', bad_file,
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Unsupported input file', result.output)

    def test_calc_rejects_invalid_date(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--overwrite', "'[excel.xlsx]'!INPUT_A=2025-01-01T00:00:00",
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Invalid scalar value', result.output)

    def test_calc_rejects_single_run_duplicate_render_key(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--render', "'[excel.xlsx]DATA'!C2=dup",
            '--render', "'[excel.xlsx]DATA'!C4=dup",
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Duplicate render key `dup`.', result.output)

    def test_calc_rejects_range_overwrite_outside_batch(self):
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--overwrite', "'[excel.xlsx]DATA'!A2:A3=1",
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Ranges can be overwritten only using `--batch`.', result.output)

    def test_calc_rejects_non_list_batch_json(self):
        batch_file = self.write_json('batch_invalid.json', {'name': 'bad'})
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--batch` JSON must be a list.', result.output)

    def test_calc_rejects_invalid_batch_overwrite_type(self):
        batch_file = self.write_json('batch_invalid_overwrite.json', [
            {'overwrite': ['bad']}
        ])
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Batch `overwrite` must be an object.', result.output)

    def test_calc_rejects_invalid_batch_outs_type(self):
        batch_file = self.write_json('batch_invalid_outs.json', [
            {'outs': {'ref': 'B1'}}
        ])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Batch `outs` must be a list.', result.output)

    def test_calc_rejects_invalid_batch_renders_type(self):
        batch_file = self.write_json('batch_invalid_renders.json', [
            {'renders': {'ref': 'B1'}}
        ])
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--batch', batch_file,
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Batch `renders` must be a list.', result.output)

    def test_calc_rejects_missing_renderable_outputs(self):
        result = self.runner.invoke(cli, [
            'calc', self.json_model,
            '--render', 'MISSING=missing',
            '--output-format', 'json',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('produced no renderable outputs', result.output)

    def test_calc_logs_missing_overwrite_refs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('formulas.cli')
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = self.runner.invoke(cli, [
                'calc', self.json_model,
                '--overwrite', 'MISSING=1',
                '--render', 'B1=result',
                '--output-format', 'json',
            ])
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Missing overwrite reference: MISSING', stream.getvalue())

    def test_calc_batch_failure_summary(self):
        batch_file = osp.join(self.tmpdir, 'batch_failure.json')
        with open(batch_file, 'w') as fh:
            json.dump([
                {
                    'name': 'ok',
                    'overwrite': {
                        "'[excel.xlsx]'!INPUT_A": 3,
                        "'[excel.xlsx]DATA'!B3": 1
                    },
                    'renders': ["'[excel.xlsx]DATA'!C2=result"]
                },
                {
                    'name': 'bad',
                    'renders': [
                        {"ref": "'[excel.xlsx]DATA'!C2", "key": 'dup'},
                        {"ref": "'[excel.xlsx]DATA'!C4", "key": 'dup'}
                    ]
                }
            ], fh)
        result = self.runner.invoke(cli, [
            'calc', self.excel,
            '--batch', batch_file,
            '--processes', '2',
            '--output-format', 'json',
            '--output-dir', self.tmpdir,
        ])
        self.assertEqual(result.exit_code, 1, result.output)
        summary = json.loads(result.output)
        self.assertEqual(['ok', 'bad'], [item['name'] for item in summary])
        self.assertEqual(['ok', 'error'], [item['status'] for item in summary])

    def test_calc_logs_missing_render_refs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('formulas.cli')
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = self.runner.invoke(cli, [
                'calc', self.json_model,
                '--render', 'MISSING=missing',
                '--output-format', 'json',
            ])
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Missing render reference: MISSING', stream.getvalue())

    def test_calc_logs_missing_refs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('formulas.cli')
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = self.runner.invoke(cli, [
                'calc', self.excel,
                '--out', 'MISSING',
                '--output-format', 'json',
            ])
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Missing output reference: MISSING', stream.getvalue())

    def test_build_full_workbook_to_file(self):
        output_file = osp.join(self.tmpdir, 'build.json')
        result = self.runner.invoke(cli, [
            'build', self.excel,
            '--output-file', output_file,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        with open(output_file) as fh:
            data = json.load(fh)
        self.assertIn("'[excel.xlsx]DATA'!C2", data)
        self.assertTrue(str(data["'[excel.xlsx]DATA'!C2"]).startswith('='))

    def test_build_mixed_inputs_stdout(self):
        result = self.runner.invoke(cli, [
            'build', self.excel, self.json_model,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertIn("'[excel.xlsx]DATA'!C2", data)
        self.assertIn('B1', data)

    def test_build_reduced_with_out(self):
        output_file = osp.join(self.tmpdir, 'reduced.json')
        full_result = self.runner.invoke(cli, ['build', self.excel])
        self.assertEqual(full_result.exit_code, 0, full_result.output)
        reduced_result = self.runner.invoke(cli, [
            'build', self.excel,
            '--out', "'[excel.xlsx]DATA'!C2",
            '--output-file', output_file,
        ])
        self.assertEqual(reduced_result.exit_code, 0, reduced_result.output)
        full_data = json.loads(full_result.output)
        with open(output_file) as fh:
            reduced_data = json.load(fh)
        self.assertIn("'[excel.xlsx]DATA'!C2", reduced_data)
        self.assertLess(len(reduced_data), len(full_data))

    def test_build_reduced_with_outs_file_roundtrip(self):
        outs_file = self.write_json('build_outs.json', ["'[excel.xlsx]DATA'!C2", 'B1'])
        built_file = osp.join(self.tmpdir, 'mixed_build.json')
        build_result = self.runner.invoke(cli, [
            'build', self.excel, self.json_model,
            '--outs', outs_file,
            '--output-file', built_file,
        ])
        self.assertEqual(build_result.exit_code, 0, build_result.output)
        calc_result = self.runner.invoke(cli, [
            'calc', built_file,
            '--render', "'[excel.xlsx]DATA'!C2=excel_result",
            '--render', 'B1=json_result',
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
            '--overwrite', "'[excel.xlsx]DATA'!B3=1",
            '--output-format', 'json',
        ])
        self.assertEqual(calc_result.exit_code, 0, calc_result.output)
        self.assertEqual({'excel_result': 10.0, 'json_result': 10}, json.loads(calc_result.output))

    def test_build_logs_missing_output_refs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('formulas.cli')
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = self.runner.invoke(cli, [
                'build', self.excel,
                '--out', 'MISSING',
                '--out', "'[excel.xlsx]DATA'!C2",
            ])
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Missing output reference: MISSING', stream.getvalue())

    def test_build_fails_when_all_outputs_missing(self):
        result = self.runner.invoke(cli, [
            'build', self.excel,
            '--out', 'MISSING',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('All requested output references are missing.', result.output)

    def test_test_defaults_to_workbook_target(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual('No differences.\n', result.output)

    def test_test_defaults_to_ods_target(self):
        ods = osp.join(mydir, 'test.ods')
        result = self.runner.invoke(cli, [
            'test', ods,
        ])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Errors(', result.output)

    def test_test_fails_without_against_for_json_only_input(self):
        result = self.runner.invoke(cli, [
            'test', self.json_model,
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('`--against` is required when no workbook inputs are provided.', result.output)

    def test_test_explicit_against(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--against', self.excel,
            '--out', "'[excel.xlsx]DATA'!C2",
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
            '--overwrite', "'[excel.xlsx]DATA'!B3=1",
        ])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Errors(', result.output)

    def test_test_scoped_outs_file(self):
        outs_file = self.write_json('test_outs.json', ["'[excel.xlsx]DATA'!C2"])
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--outs', outs_file,
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual('No differences.\n', result.output)

    def test_test_failure_with_overwrite(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
        ])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Errors(', result.output)

    def test_test_summary_table(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--out', "'[excel.xlsx]DATA'!C2",
            '--summary',
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('status', result.output)
        self.assertIn('PASS', result.output)
        self.assertIn('scoped', result.output)
        self.assertIn('No differences.', result.output)

    def test_test_logs_missing_overwrite_refs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('formulas.cli')
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = self.runner.invoke(cli, [
                'test', self.excel,
                '--overwrite', 'MISSING=1',
            ])
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Missing overwrite reference: MISSING', stream.getvalue())

    def test_test_fails_when_all_outputs_missing(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--out', 'MISSING',
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('All requested output references are missing.', result.output)

    def test_test_summary_fail(self):
        result = self.runner.invoke(cli, [
            'test', self.excel,
            '--overwrite', "'[excel.xlsx]'!INPUT_A=3",
            '--summary',
        ])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('FAIL', result.output)
        self.assertIn('missing_overwrites', result.output)

    def test_serve_health_endpoint(self):
        _set_serve_environment(
            _normalize_serve_config((self.excel,), '127.0.0.1', 5000, False)
        )
        response = create_app().test_client().get('/api/health')
        self.assertEqual(200, response.status_code)
        self.assertEqual({'status': 'ok'}, response.get_json())

    def test_serve_model_endpoint(self):
        _set_serve_environment(
            _normalize_serve_config((self.excel, self.json_model), '127.0.0.1', 5000, False)
        )
        response = create_app().test_client().get('/api/model')
        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertIn(self.excel, data['files'])
        self.assertIn(self.json_model, data['files'])
        self.assertGreater(data['nodes'], 0)

    def test_serve_calculate_endpoint(self):
        _set_serve_environment(
            _normalize_serve_config((self.excel,), '127.0.0.1', 5000, False)
        )
        response = create_app().test_client().post('/api/calculate', json={
            'inputs': {
                "'[excel.xlsx]'!INPUT_A": 3,
                "'[excel.xlsx]DATA'!B3": 1
            },
            'renders': ["'[excel.xlsx]DATA'!C2=result"]
        })
        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertEqual({'result': 10.0}, data['outputs'])
        self.assertEqual([], data['warnings'])

    def test_serve_calculate_endpoint_validation(self):
        _set_serve_environment(
            _normalize_serve_config((self.excel,), '127.0.0.1', 5000, False)
        )
        response = create_app().test_client().post('/api/calculate', json={'inputs': []})
        self.assertEqual(400, response.status_code)
        self.assertIn('`inputs` must be an object.', response.get_json()['error'])

    def test_serve_gui_routes(self):
        _set_serve_environment(
            _normalize_serve_config((self.excel,), '127.0.0.1', 5000, False)
        )
        client = create_app().test_client()
        self.assertEqual(200, client.get('/').status_code)
        page = client.get('/').get_data(as_text=True)
        self.assertIn("fetch('/api/calculate'", page)

    def test_serve_command_invokes_schedula_runner(self):
        with patch('schedula.utils.form.cli.run.invoke') as run_mock:
            result = self.runner.invoke(cli, [
                'serve', self.excel,
                '--host', '0.0.0.0',
                '--port', '5050',
            ])
        self.assertEqual(0, result.exit_code, result.output)
        run_mock.assert_called_once()
        self.assertEqual('formulas.app:create_app()', run_mock.call_args[0][0].params['app'])
        self.assertEqual(json.dumps((self.excel,)), os.environ['FORMULAS_SERVE_FILES'])
        self.assertEqual('false', os.environ['FORMULAS_SERVE_CIRCULAR'])
