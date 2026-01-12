defmodule ReqLLM.Providers.QwenEmbed do
  @moduledoc """
  Provider for Qwen3-VL-Embedding multimodal embedding server.

  Supports text, video, and image embeddings via OpenAI-compatible `/v1/embeddings` endpoint.
  Extends the standard OpenAI format to support multimodal inputs.

  ## Server API Format (OpenAI-compatible with multimodal extension)

  **Endpoint:** POST /v1/embeddings

  **Request (text only - standard OpenAI):**
  ```json
  {
    "model": "Qwen3-VL-Embedding-8B",
    "input": "query text"
  }
  ```

  **Request (multimodal - extended format):**
  ```json
  {
    "model": "Qwen3-VL-Embedding-8B",
    "input": [
      {"text": "query text"},
      {"video": "https://..."},
      {"image": "https://..."}
    ]
  }
  ```

  **Response (standard OpenAI):**
  ```json
  {
    "data": [{"embedding": [0.1, -0.2, ...], "index": 0}],
    "model": "Qwen3-VL-Embedding-8B"
  }
  ```

  ## Usage

      # Text embedding (standard)
      {:ok, embedding} = ReqLLM.embed("qwen_embed:Qwen3-VL-Embedding-8B", "query text")

      # Video embedding with ContentPart
      alias ReqLLM.Message.ContentPart
      {:ok, embedding} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        ContentPart.video_url("https://example.com/video.mp4"),
        base_url: "http://localhost:8003"
      )

      # Image embedding
      {:ok, embedding} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        ContentPart.image_url("https://example.com/image.jpg"),
        base_url: "http://localhost:8003"
      )

      # Multimodal batch
      {:ok, embeddings} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        [
          ContentPart.video_url("https://example.com/video.mp4"),
          ContentPart.text("person sitting at desk")
        ],
        base_url: "http://localhost:8003"
      )

  ## Configuration

      # Set base_url in config or per-call
      config :req_llm, :qwen_embed,
        base_url: "http://localhost:8003"
  """

  use ReqLLM.Provider,
    id: :qwen_embed,
    default_base_url: "http://localhost:8003",
    default_env_key: "QWEN_EMBED_API_KEY"

  alias ReqLLM.Message.ContentPart

  @default_receive_timeout 120_000

  @impl ReqLLM.Provider
  def prepare_request(:embedding, model_spec, input, opts) do
    prepare_multimodal_embedding_request(model_spec, input, opts)
  end

  def prepare_request(operation, _model_spec, _input, _opts) do
    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter: "operation: #{inspect(operation)} not supported by qwen_embed provider"
     )}
  end

  defp prepare_multimodal_embedding_request(model_spec, input, opts) do
    with {:ok, model} <- ReqLLM.model(model_spec) do
      base_url = Keyword.get(opts, :base_url, default_base_url())
      receive_timeout = Keyword.get(opts, :receive_timeout, @default_receive_timeout)
      http_opts = Keyword.get(opts, :req_http_options, [])

      normalized_input = normalize_embedding_input(input)

      request =
        Req.new(
          [
            url: "/v1/embeddings",
            method: :post,
            base_url: base_url,
            receive_timeout: receive_timeout
          ] ++ http_opts
        )
        |> Req.Request.register_options([:model, :input, :operation])
        |> Req.Request.merge_options(
          model: model.id,
          input: normalized_input,
          operation: :embedding
        )
        |> attach(model, opts)

      {:ok, request}
    end
  end

  # Single text string - keep as-is (standard OpenAI format)
  defp normalize_embedding_input(input) when is_binary(input), do: input

  # List of text strings - keep as-is (standard OpenAI format)
  defp normalize_embedding_input(inputs) when is_list(inputs) and is_binary(hd(inputs)), do: inputs

  # ContentPart text - convert to string for standard format
  defp normalize_embedding_input(%ContentPart{type: :text, text: text}), do: text

  # ContentPart video - convert to multimodal format
  defp normalize_embedding_input(%ContentPart{type: :video_url, url: url}) do
    [%{video: url}]
  end

  # ContentPart image URL - convert to multimodal format
  defp normalize_embedding_input(%ContentPart{type: :image_url, url: url}) do
    [%{image: url}]
  end

  # ContentPart image binary - convert to base64 data URL
  defp normalize_embedding_input(%ContentPart{type: :image, data: data, media_type: media_type}) do
    base64 = Base.encode64(data)
    [%{image: "data:#{media_type};base64,#{base64}"}]
  end

  # List of ContentParts or mixed inputs - convert to multimodal format
  defp normalize_embedding_input(inputs) when is_list(inputs) do
    Enum.map(inputs, &normalize_single_input/1)
  end

  # Map inputs (already in multimodal format)
  defp normalize_embedding_input(%{text: _} = input), do: [input]
  defp normalize_embedding_input(%{video: _} = input), do: [input]
  defp normalize_embedding_input(%{image: _} = input), do: [input]

  defp normalize_single_input(input) when is_binary(input), do: %{text: input}
  defp normalize_single_input(%ContentPart{type: :text, text: text}), do: %{text: text}
  defp normalize_single_input(%ContentPart{type: :video_url, url: url}), do: %{video: url}
  defp normalize_single_input(%ContentPart{type: :image_url, url: url}), do: %{image: url}

  defp normalize_single_input(%ContentPart{type: :image, data: data, media_type: media_type}) do
    base64 = Base.encode64(data)
    %{image: "data:#{media_type};base64,#{base64}"}
  end

  defp normalize_single_input(%{text: _} = input), do: input
  defp normalize_single_input(%{video: _} = input), do: input
  defp normalize_single_input(%{image: _} = input), do: input
  defp normalize_single_input(%{"text" => t}), do: %{text: t}
  defp normalize_single_input(%{"video" => v}), do: %{video: v}
  defp normalize_single_input(%{"image" => i}), do: %{image: i}

  @impl ReqLLM.Provider
  def attach(request, _model_input, opts) do
    api_key = opts[:api_key]

    request =
      if api_key do
        Req.Request.put_header(request, "authorization", "Bearer #{api_key}")
      else
        request
      end

    request
    |> Req.Request.put_header("content-type", "application/json")
    |> Req.Request.append_request_steps(encode: &encode_body/1)
    |> Req.Request.append_response_steps(decode: &decode_response/1)
  end

  @impl ReqLLM.Provider
  def encode_body(request) do
    model = request.options[:model]
    input = request.options[:input]

    body = %{model: model, input: input}
    encoded_body = Jason.encode!(body)

    request
    |> Map.put(:body, encoded_body)
  end

  @impl ReqLLM.Provider
  def decode_response({req, %Req.Response{status: 200, body: body} = resp}) do
    parsed_body = ensure_parsed_body(body)
    {req, %{resp | body: parsed_body}}
  end

  def decode_response({req, %Req.Response{status: status, body: body} = _resp}) do
    {req,
     ReqLLM.Error.API.Response.exception(
       reason: "Qwen Embed API error",
       status: status,
       response_body: body
     )}
  end

  def decode_response({req, resp}), do: {req, resp}

  defp ensure_parsed_body(body) when is_binary(body), do: Jason.decode!(body)
  defp ensure_parsed_body(body), do: body

  @impl ReqLLM.Provider
  def extract_usage(_body, _model) do
    {:ok, %{input_tokens: 0, output_tokens: 0, total_tokens: 0}}
  end
end
