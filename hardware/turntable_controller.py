import serial
import serial.tools.list_ports
import time
import threading

class TurntableController:
    def __init__(self, port=None, baudrate=115200, timeout=2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.lock = threading.Lock()
        
    def connect(self):
        with self.lock:
            if self.serial and self.serial.is_open:
                return True
                
            if not self.port:
                # Auto-detect an ESP32 or generic serial device
                ports = serial.tools.list_ports.comports()
                for p in ports:
                    # Common identifiers for ESP32 boards (CP210x, CH340, etc.)
                    if "CP210" in p.description or "CH340" in p.description or "Serial" in p.description:
                        self.port = p.device
                        break
                
                if not self.port:
                    raise RuntimeError("Could not auto-detect ESP32 serial port.")
                    
            try:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                # Wait for ESP32 to reboot and print READY
                time.sleep(2) 
                self.serial.reset_input_buffer()
                return True
            except Exception as e:
                raise RuntimeError(f"Failed to connect to turntable on {self.port}: {e}")

    def disconnect(self):
        with self.lock:
            if self.serial and self.serial.is_open:
                self.serial.close()
                self.serial = None

    def _send_command(self, cmd, wait_for="OK"):
        self.connect()
        with self.lock:
            self.serial.write(f"{cmd}\n".encode('ascii'))
            self.serial.flush()
            
            # Wait for expected response
            start_time = time.time()
            while (time.time() - start_time) < self.timeout:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('ascii', errors='ignore').strip()
                    if line.startswith(wait_for) or line.startswith("ERROR"):
                        return line
                time.sleep(0.01)
            
            raise TimeoutError(f"Command '{cmd}' timed out waiting for '{wait_for}'")

    def rotate(self, angle_deg, block=True):
        """Rotates the turntable by angle_deg. If block=True, waits until rotation is complete."""
        # Send ROTATE command and wait for initial "OK MOVING"
        resp = self._send_command(f"ROTATE {angle_deg}", wait_for="OK")
        if "ERROR" in resp:
            raise RuntimeError(f"Turntable error: {resp}")
            
        if block:
            # Wait for "OK DONE"
            self.wait_until_idle()

    def wait_until_idle(self, timeout=30.0):
        """Blocks until the turntable reports IDLE or DONE."""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            status = self.get_status()
            if status['state'] == 'IDLE':
                return True
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for turntable to become idle.")

    def set_speed(self, speed):
        resp = self._send_command(f"SET_SPEED {speed}", wait_for="OK")
        if "ERROR" in resp:
            raise RuntimeError(f"Turntable error: {resp}")

    def stop(self):
        resp = self._send_command("STOP", wait_for="OK")
        if "ERROR" in resp:
            raise RuntimeError(f"Turntable error: {resp}")

    def calibrate(self):
        resp = self._send_command("CALIBRATE", wait_for="OK")
        if "ERROR" in resp:
            raise RuntimeError(f"Turntable error: {resp}")

    def get_status(self):
        resp = self._send_command("STATUS", wait_for="OK")
        # Example format: "OK MOVING 45.00" or "OK IDLE 90.00"
        parts = resp.split()
        if len(parts) >= 3 and parts[0] == "OK":
            return {
                "state": parts[1],
                "angle": float(parts[2])
            }
        raise RuntimeError(f"Failed to parse status: {resp}")

# Global singleton instance for the FastAPI backend
turntable = TurntableController()
