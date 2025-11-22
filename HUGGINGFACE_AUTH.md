# HuggingFace Authentication Setup

This guide explains how to set up HuggingFace authentication for accessing private or gated models.

## Why Do I Need This?

Some models on HuggingFace require authentication:
- **Private models**: Models that are not publicly accessible
- **Gated models**: Models that require agreeing to terms of use before access
- **Examples**: `casperhansen/qwen2.5-14b-instruct-awq`, Meta's Llama models, etc.

## Error Symptoms

If you see this error, you need to set up authentication:
```
401 Client Error: Unauthorized for url: https://huggingface.co/...
RepositoryNotFoundError: Invalid username or password.
```

## Quick Setup (3 Steps)

### 1. Get Your HuggingFace Token

1. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., "Aeon-AI-vLLM")
4. Select **"Read"** access (sufficient for model downloads)
5. Click **"Generate token"**
6. **Copy the token** (starts with `hf_...`)

### 2. Create Your `.env` File

```bash
# From the Aeon-AI repository root
cp .env.template .env
```

### 3. Add Your Token to `.env`

Open `.env` and uncomment/edit the `HF_TOKEN` line:

```bash
# HuggingFace Hub Token (required for private/gated models)
HF_TOKEN=hf_your_actual_token_here
```

**Important**:
- Replace `hf_your_actual_token_here` with your actual token
- Keep this file secret! It's already in `.gitignore` to prevent accidental commits

## Verify Setup

After setting up your token:

1. **Stop any running vLLM servers**:
   ```bash
   ./inference/stop_vllm.sh
   # Or: tmux kill-session -t aeon-vllm
   ```

2. **Start vLLM** (it will load the token from `.env`):
   ```bash
   ./inference/start_vllm.sh
   ```

3. **Check the logs** to verify authentication worked:
   ```bash
   tail -f ./inference/vllm.log
   ```

You should see the model loading without authentication errors.

## Troubleshooting

### Token Not Working?

1. **Verify token is correct**: Check for copy-paste errors (no extra spaces)
2. **Regenerate token**: Create a new token on HuggingFace and update `.env`
3. **Check model access**: Some gated models require accepting terms first
   - Visit the model page on HuggingFace (e.g., `huggingface.co/casperhansen/qwen2.5-14b-instruct-awq`)
   - Click "Agree" to the terms if prompted

### Still Getting 401 Errors?

1. **Ensure `.env` is in the right place**: Must be in repository root (`Aeon-AI/.env`)
2. **Check file permissions**: `.env` should be readable
   ```bash
   ls -la .env
   ```
3. **Restart vLLM completely**:
   ```bash
   pkill -9 -f 'vllm serve'
   ./inference/start_vllm.sh
   ```

### Model is Truly Private?

If you don't have access to the model:
1. **Request access** from the model owner on HuggingFace
2. **Use an alternative model**: Check `.env.template` for public model options
   - Example: `MODEL_NAME=Qwen/Qwen2.5-14B-Instruct` (public, no token needed)

## Security Best Practices

✅ **DO**:
- Keep your `.env` file local (it's in `.gitignore`)
- Use "Read" tokens for model downloads (not "Write")
- Rotate tokens periodically

❌ **DON'T**:
- Commit `.env` to git
- Share your token publicly
- Use tokens with more permissions than needed

## Next Steps

After authentication is working:
1. Configure your preferred model in `.env`
2. Adjust GPU memory settings if needed
3. Start building with Aeon-AI!

For more configuration options, see `.env.template` with detailed comments.
