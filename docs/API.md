# Claude-Rust Inference API

This document describes the REST API exposed by the `inference` server. All endpoints accept and return JSON.

## Base URL

Standard development server runs at: `http://localhost:8000`

## Endpoints

### 1. Generate Text (`POST /generate`)

Generates a completion based on the provided prompt and sampling parameters.

**Request Body (`application/json`)**:

```json
{
  "prompt": "Write a Rust function to solve failing tests.",
  "max_new_tokens": 128,      // (Optional) Max tokens to generate
  "temperature": 0.7,         // (Optional) Creativity
  "top_k": 40,                // (Optional) Token sampling
  "top_p": 0.9,               // (Optional) Nucleus sampling
  "stop_sequences": [         // (Optional) Strings that halt generation
    "\n\n", "User:"
  ],
  "do_sample": true           // (Optional) Set false for greedy decoding
}
```

**Response (`200 OK`)**:

```json
{
  "text": "Here is a Rust function to fix failing tests:\n\nfn fix_tests(code: &str) -> String {\n    // ... implementation\n}",
  "finish_reason": "length"   // "length", "stop", or "eos"
}
```

### 2. Tokenize (`POST /tokenize`)

Encodes text into token IDs. Useful for client-side length calculation.

**Request**:
```json
{ "text": "Hello world" }
```

**Response**:
```json
{ "ids": [15496, 995] }
```

### 3. Detokenize (`POST /detokenize`)

Decodes token IDs back to text.

**Request**:
```json
{ "ids": [15496, 995] }
```

**Response**:
```json
{ "text": "Hello world" }
```

### 4. Health Check (`GET /health`)

Verifies the server is running and model is loaded.

**Response**:
```json
{ "status": "ok", "model": "claude-rust-small" }
```

## Streaming (Optional - if implemented)

### Generate Stream (`POST /generate_stream`)

Returns a Server-Sent Events (SSE) stream of tokens as they are generated.

**Request**: Same as `/generate`.

**Response (SSE)**:
```
data: {"token": "H", "id": 35}
data: {"token": "e", "id": 68}
...
event: done
data: [DONE]
```

## Error Handling

Standard HTTP status codes are used:

*   `400 Bad Request`: Invalid JSON or parameters (e.g., negative temperature).
*   `404 Not Found`: Model or resource unavailable.
*   `500 Internal Server Error`: Backend/CUDA error or crash.
