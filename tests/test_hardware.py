import pytest
import sys
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.turntable_controller import TurntableController

def test_turntable_controller_status():
    # Mock pyserial so we don't need a real device connected
    with patch('serial.Serial') as mock_serial:
        mock_instance = MagicMock()
        mock_serial.return_value = mock_instance
        
        # Simulate ESP32 responding to STATUS command
        mock_instance.in_waiting = 1
        mock_instance.readline.return_value = b"OK IDLE 45.00\n"
        
        controller = TurntableController(port="COM_TEST")
        
        # Test get_status()
        status = controller.get_status()
        
        assert status["state"] == "IDLE"
        assert status["angle"] == 45.0
        
def test_turntable_controller_rotate():
    with patch('serial.Serial') as mock_serial:
        mock_instance = MagicMock()
        mock_serial.return_value = mock_instance
        
        # Simulate ESP32 responding to ROTATE command
        mock_instance.in_waiting = 1
        mock_instance.readline.return_value = b"OK MOVING\n"
        
        controller = TurntableController(port="COM_TEST")
        
        # Override wait_until_idle to not block indefinitely in test
        controller.wait_until_idle = MagicMock()
        
        controller.rotate(90.0, block=True)
        
        # Verify the command was sent correctly
        mock_instance.write.assert_called_with(b"ROTATE 90.0\n")
        assert controller.wait_until_idle.called
