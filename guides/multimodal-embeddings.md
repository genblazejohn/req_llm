# Multimodal Embeddings with ReqLLM

This guide covers how to generate embeddings for text, video, and images using ReqLLM, and how to compare them using cosine similarity.

## Setup

Add to your dependencies:

```elixir
{:req_llm, github: "genblazejohn/req_llm", branch: "video-url-support"}
```

## Basic Text Embeddings (OpenAI)

```elixir
# Single text
{:ok, embedding} = ReqLLM.embed("openai:text-embedding-3-small", "hello world")
# Returns: {:ok, [0.123, -0.456, ...]}  # 1536-dimensional vector

# Batch texts
{:ok, embeddings} = ReqLLM.embed("openai:text-embedding-3-small", ["hello", "world"])
# Returns: {:ok, [[...], [...]]}  # List of vectors
```

## Multimodal Embeddings (Qwen)

The `qwen_embed` provider supports text, video, and image embeddings via a self-hosted Qwen3-VL-Embedding server.

```elixir
alias ReqLLM.Message.ContentPart

# Configure base URL for your Qwen server
opts = [base_url: "http://localhost:8003"]

# Text embedding
{:ok, text_emb} = ReqLLM.embed(
  "qwen_embed:Qwen3-VL-Embedding-8B",
  ContentPart.text("person sitting at desk"),
  opts
)

# Video embedding
{:ok, video_emb} = ReqLLM.embed(
  "qwen_embed:Qwen3-VL-Embedding-8B",
  ContentPart.video_url("https://example.com/video.mp4"),
  opts
)

# Image embedding
{:ok, image_emb} = ReqLLM.embed(
  "qwen_embed:Qwen3-VL-Embedding-8B",
  ContentPart.image_url("https://example.com/image.jpg"),
  opts
)

# Batch multimodal (mix of types)
{:ok, embeddings} = ReqLLM.embed(
  "qwen_embed:Qwen3-VL-Embedding-8B",
  [
    ContentPart.video_url("https://example.com/video.mp4"),
    ContentPart.text("person sitting at desk"),
    ContentPart.image_url("https://example.com/image.jpg")
  ],
  opts
)
```

## Cosine Similarity

ReqLLM doesn't include a built-in similarity function. Here's how to implement it:

```elixir
defmodule EmbeddingSimilarity do
  @doc """
  Calculate cosine similarity between two embedding vectors.
  Returns a value between -1 and 1, where 1 = identical.
  """
  def cosine_similarity(vec1, vec2) when length(vec1) == length(vec2) do
    dot = Enum.zip(vec1, vec2) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()
    norm1 = :math.sqrt(Enum.map(vec1, &(&1 * &1)) |> Enum.sum())
    norm2 = :math.sqrt(Enum.map(vec2, &(&1 * &1)) |> Enum.sum())
    dot / (norm1 * norm2)
  end

  @doc """
  Find most similar items from a list of embeddings.
  Returns list of {index, similarity} sorted by similarity descending.
  """
  def find_similar(query_embedding, embeddings, top_k \\ 5) do
    embeddings
    |> Enum.with_index()
    |> Enum.map(fn {emb, idx} -> {idx, cosine_similarity(query_embedding, emb)} end)
    |> Enum.sort_by(fn {_idx, sim} -> sim end, :desc)
    |> Enum.take(top_k)
  end
end
```

## Complete Example: Video Search

```elixir
alias ReqLLM.Message.ContentPart

defmodule VideoSearch do
  @opts [base_url: "http://localhost:8003"]
  @model "qwen_embed:Qwen3-VL-Embedding-8B"

  def index_videos(video_urls) do
    # Generate embeddings for all videos
    {:ok, embeddings} = ReqLLM.embed(
      @model,
      Enum.map(video_urls, &ContentPart.video_url/1),
      @opts
    )

    # Return map of url -> embedding
    Enum.zip(video_urls, embeddings) |> Map.new()
  end

  def search_by_text(query_text, video_index) do
    # Embed the text query
    {:ok, query_emb} = ReqLLM.embed(
      @model,
      ContentPart.text(query_text),
      @opts
    )

    # Compare against all videos
    video_index
    |> Enum.map(fn {url, emb} ->
      {url, EmbeddingSimilarity.cosine_similarity(query_emb, emb)}
    end)
    |> Enum.sort_by(fn {_url, sim} -> sim end, :desc)
  end

  def search_by_image(image_url, video_index) do
    {:ok, query_emb} = ReqLLM.embed(
      @model,
      ContentPart.image_url(image_url),
      @opts
    )

    video_index
    |> Enum.map(fn {url, emb} ->
      {url, EmbeddingSimilarity.cosine_similarity(query_emb, emb)}
    end)
    |> Enum.sort_by(fn {_url, sim} -> sim end, :desc)
  end
end
```

### Usage

```elixir
videos = ["https://example.com/v1.mp4", "https://example.com/v2.mp4"]
index = VideoSearch.index_videos(videos)

# Search by text
results = VideoSearch.search_by_text("person cooking in kitchen", index)
# => [{"https://example.com/v1.mp4", 0.87}, {"https://example.com/v2.mp4", 0.34}]

# Search by image
results = VideoSearch.search_by_image("https://example.com/query.jpg", index)
```

## Supported ContentPart Types for Embedding

| Type | Constructor | Example |
|------|-------------|---------|
| Text | `ContentPart.text(string)` | `ContentPart.text("hello")` |
| Video URL | `ContentPart.video_url(url)` | `ContentPart.video_url("https://...")` |
| Image URL | `ContentPart.image_url(url)` | `ContentPart.image_url("https://...")` |
| Image binary | `ContentPart.image(binary, mime)` | `ContentPart.image(data, "image/png")` |

## Error Handling

```elixir
case ReqLLM.embed("qwen_embed:model", ContentPart.video_url(url), opts) do
  {:ok, embedding} ->
    # Use embedding

  {:error, %ReqLLM.Error.API.Response{} = error} ->
    # API error (server returned error)
    Logger.error("API error: #{Exception.message(error)}")

  {:error, %ReqLLM.Error.Invalid.Parameter{} = error} ->
    # Invalid input (empty text, unsupported type, etc.)
    Logger.error("Invalid input: #{Exception.message(error)}")

  {:error, error} ->
    # Network or other error
    Logger.error("Error: #{inspect(error)}")
end
```

## Qwen Embedding Server API

The `qwen_embed` provider expects a server with the following API:

**Endpoint:** `POST /embed`

**Request:**
```json
{
  "inputs": [
    {"text": "query text"},
    {"video": "https://..."},
    {"image": "https://..."}
  ]
}
```

**Response:**
```json
{
  "data": [
    {"embedding": [0.1, -0.2, ...], "index": 0},
    {"embedding": [0.3, 0.4, ...], "index": 1}
  ],
  "model": "Qwen3-VL-Embedding-8B"
}
```
