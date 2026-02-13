import subprocess
import json


def execute_python_code(code: str, context: dict = dict(), output_string_limit: int = 512) -> str:
    url = "http://127.0.0.1:8001/execute"
    payload = json.dumps({
        "code": code
    })

    try:
        cmd = [
            "curl", "-s", 
            "-X", "POST", url,
            "-H", "Content-Type: application/json",
            "-d", payload
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        try:
            response_data = json.loads(result.stdout)
            output = response_data.get("output", result.stdout)
            new_context =response_data.get("context", result.stdout)
            context.update(new_context)
            
            if len(output) > output_string_limit:
                return output[:output_string_limit] + "...(max tool output length exceeded)\n-------------------------------\n", context
            else:
                return output, context
        except json.JSONDecodeError:
            return f"Error: Server returned invalid JSON.\nRaw response: {result.stdout}", context

    except subprocess.CalledProcessError as e:
        return f"Error: curl command failed (Exit Code: {e.returncode}).\nStderr: {e.stderr}", context
    except FileNotFoundError:
        return "Error: 'curl' command not found on this system.", context
    except Exception as e:
        return f"Error: {str(e)}", context
    
description = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code in a sandboxed environment.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": ""
                    }
                },
                "required": ["code"]
            }
        }
    }
]