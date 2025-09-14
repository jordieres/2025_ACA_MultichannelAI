# 🧠 Running Ollama Locally on GPU

This guide explains how to run an **Ollama server locally on GPUs** and ensure that it stays running in the background, even if you close your terminal or disconnect from VSCode.

---

## 1. Configure environment variables

Decide where models will be stored and which GPUs to use. You can export them in your shell or directly in the service (see below).

```bash
export OLLAMA_MODELS="$HOME/.ollama/models"
export OLLAMA_HOST="http://127.0.0.1:11500"
export CUDA_VISIBLE_DEVICES=0,1        # adapt to your GPUs
export OLLAMA_KEEP_ALIVE=30m
```

- `OLLAMA_MODELS` → path to store models  
- `OLLAMA_HOST` → server binding address and port  
- `CUDA_VISIBLE_DEVICES` → which GPUs Ollama will use  
- `OLLAMA_KEEP_ALIVE` → how long to keep models loaded in memory after requests  

---

## 2. Create a systemd user service

Create the service file:

```bash
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/ollama.service
```

Paste the following:

```ini
[Unit]
Description=Ollama server
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_HOST=http://127.0.0.1:11500"
Environment="OLLAMA_MODELS=%h/.ollama/models"
WorkingDirectory=%h
StandardOutput=append:%h/ollama.log
StandardError=append:%h/ollama.log

[Install]
WantedBy=default.target
```

---

## 3. Enable and start the service

Reload systemd and enable the Ollama service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ollama.service
```

Check the status:

```bash
systemctl --user status ollama.service
```

Follow logs in real time:

```bash
journalctl --user -u ollama.service -f
```

---

## 4. Verify GPU usage

Run:

```bash
nvidia-smi
```

You should see `/usr/local/bin/ollama` using GPU memory.  
Multiple Ollama processes may appear if several models are loaded in parallel.

---

## 5. Connect from Python

Point your Python client to the configured host:

```python
import ollama

client = ollama.Client(host="http://127.0.0.1:11500")

response = client.chat(
    model="llama3:8b",
    messages=[{"role": "user", "content": "Hello Ollama!"}]
)

print(response.message.content)
```

---

✅ Ollama is now running permanently in the background on your GPUs.  
✅ Python code connects directly through the configured host and port.  
✅ Models remain cached in GPU memory for faster inference.  

---
