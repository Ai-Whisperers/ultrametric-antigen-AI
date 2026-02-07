"""Tests for CLI functionality."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module under test
from hiv_analysis import cli


class TestCLI:
    """Test cases for the CLI module."""

    def test_main_help_command(self, capsys):
        """Test that help command works."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'help']):
            result = cli.main()
        
        assert result == 0
        captured = capsys.readouterr()
        assert "HIV Sequence Analysis Pipeline" in captured.out

    def test_main_setup_command(self):
        """Test setup command execution."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'setup']):
            with patch('hiv_analysis.scripts.setup_hiv_data.main') as mock_setup:
                mock_setup.return_value = 0
                result = cli.main()
        
        assert result == 0
        mock_setup.assert_called_once()

    def test_commands_requiring_input_file(self, capsys):
        """Test that commands requiring input file fail without it."""
        commands = ['align', 'score', 'view', 'export']
        
        for command in commands:
            with patch.object(sys, 'argv', ['hiv-analysis', command]):
                with pytest.raises(SystemExit):
                    cli.main()

    def test_align_command_with_input(self, capsys):
        """Test align command with input file."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'align', 'test.fasta']):
            result = cli.main()
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Aligning sequences: test.fasta" in captured.out

    def test_score_command_with_input(self, capsys):
        """Test score command with input file."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'score', 'test.fasta']):
            result = cli.main()
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Scoring conservation: test.fasta" in captured.out

    def test_view_command_with_input(self, capsys):
        """Test view command with input file."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'view', 'test.fasta']):
            result = cli.main()
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Creating visualization: test.fasta" in captured.out

    def test_export_command_with_input(self, capsys):
        """Test export command with input file."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'export', 'test.fasta']):
            result = cli.main()
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Exporting formats: test.fasta" in captured.out

    def test_exception_handling(self, capsys):
        """Test that exceptions are handled gracefully."""
        with patch.object(sys, 'argv', ['hiv-analysis', 'setup']):
            with patch('hiv_analysis.scripts.setup_hiv_data.main') as mock_setup:
                mock_setup.side_effect = Exception("Test error")
                result = cli.main()
        
        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err