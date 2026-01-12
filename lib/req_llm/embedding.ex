defmodule ReqLLM.Embedding do
  @moduledoc """
  Embedding functionality for ReqLLM.

  This module provides embedding generation capabilities with support for:
  - Single text embedding generation
  - Batch text embedding generation
  - Multimodal embedding generation (text, image, video via ContentPart)
  - Model validation for embedding support

  ## Multimodal Embeddings

  Some providers (like Qwen) support multimodal embeddings. Use `ContentPart` to
  embed images and videos:

      alias ReqLLM.Message.ContentPart

      # Video embedding
      {:ok, embedding} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        ContentPart.video_url("https://example.com/video.mp4"),
        base_url: "http://localhost:8003"
      )

      # Image embedding
      {:ok, embedding} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        ContentPart.image_url("https://example.com/image.jpg")
      )

      # Batch multimodal
      {:ok, embeddings} = ReqLLM.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        [
          ContentPart.video_url("https://example.com/video.mp4"),
          ContentPart.text("person sitting at desk")
        ]
      )
  """

  alias LLMDB.Model
  alias ReqLLM.Message.ContentPart

  # Get embedding models dynamically from LLMDB
  defp get_embedding_models do
    ReqLLM.Providers.list()
    |> Enum.flat_map(fn provider ->
      LLMDB.models(provider)
      |> Enum.filter(fn model ->
        model.capabilities && model.capabilities.embeddings
      end)
      |> Enum.map(fn model ->
        LLMDB.Model.spec(model)
      end)
    end)
  end

  @base_schema NimbleOptions.new!(
                 dimensions: [
                   type: :pos_integer,
                   doc: "Number of dimensions for the embedding vector"
                 ],
                 encoding_format: [
                   type: {:in, ["float", "base64"]},
                   doc: "Format for encoding the embedding vector",
                   default: "float"
                 ],
                 user: [
                   type: :string,
                   doc: "User identifier for tracking and abuse detection"
                 ],
                 provider_options: [
                   type: {:or, [:map, {:list, :any}]},
                   doc: "Provider-specific options (keyword list or map)",
                   default: []
                 ],
                 req_http_options: [
                   type: {:or, [:map, {:list, :any}]},
                   doc: "Req-specific options (keyword list or map)",
                   default: []
                 ],
                 fixture: [
                   type: {:or, [:string, {:tuple, [:atom, :string]}]},
                   doc: "HTTP fixture for testing (provider inferred from model if string)"
                 ]
               )

  @doc """
  Returns the list of supported embedding model specifications.

  ## Examples

      ReqLLM.Embedding.supported_models()
      #=> ["openai:text-embedding-3-small", "openai:text-embedding-3-large", "openai:text-embedding-ada-002", "google:gemini-embedding-001"]

  """
  @spec supported_models() :: [String.t()]
  def supported_models, do: get_embedding_models()

  @doc """
  Validates that a model supports embedding operations.

  ## Parameters

    * `model_spec` - Model specification in various formats

  ## Examples

      ReqLLM.Embedding.validate_model("openai:text-embedding-3-small")
      #=> {:ok, %LLMDB.Model{provider: :openai, model: "text-embedding-3-small"}}

      ReqLLM.Embedding.validate_model("anthropic:claude-3-sonnet")
      #=> {:error, :embedding_not_supported}

  """
  @spec validate_model(String.t() | {atom(), keyword()} | struct()) ::
          {:ok, Model.t()} | {:error, term()}
  def validate_model(model_spec) do
    with {:ok, model} <- ReqLLM.model(model_spec) do
      model_string = LLMDB.Model.spec(model)
      embedding_models = get_embedding_models()

      cond do
        # Model is in LLMDB embedding models list
        model_string in embedding_models ->
          validate_provider_embedding_support(model, model_string)

        # Model has embedding capability flag set (custom providers)
        model.capabilities && model.capabilities.embeddings ->
          validate_provider_embedding_support(model, model_string)

        # Model doesn't support embeddings
        true ->
          {:error,
           ReqLLM.Error.Invalid.Parameter.exception(
             parameter: "model: #{model_string} does not support embedding operations"
           )}
      end
    end
  end

  defp validate_provider_embedding_support(model, model_string) do
    case ReqLLM.provider(model.provider) do
      {:ok, provider_module} ->
        case provider_module.prepare_request(:embedding, model, "test", []) do
          {:ok, _} ->
            {:ok, model}

          {:error, _} ->
            {:error,
             ReqLLM.Error.Invalid.Parameter.exception(
               parameter:
                 "model: #{model_string} provider does not support embedding operations"
             )}
        end

      {:error, _} ->
        {:error,
         ReqLLM.Error.Invalid.Parameter.exception(
           parameter: "model: #{model_string} provider not found"
         )}
    end
  end

  @doc """
  Returns the base embedding options schema.

  This schema contains embedding-specific options that are vendor-neutral.
  """
  @spec schema :: NimbleOptions.t()
  def schema, do: @base_schema

  @doc """
  Generates embeddings for single or multiple inputs.

  Accepts text strings, ContentPart structs, or lists of either, automatically
  handling both cases using pattern matching.

  ## Parameters

    * `model_spec` - Model specification in various formats
    * `input` - Text string, ContentPart, or list of either
    * `opts` - Additional options (keyword list)

  ## Options

    * `:dimensions` - Number of dimensions for embeddings
    * `:encoding_format` - Format for encoding ("float" or "base64")
    * `:user` - User identifier for tracking
    * `:provider_options` - Provider-specific options
    * `:base_url` - Override the provider's default base URL

  ## Examples

      # Single text input
      {:ok, embedding} = ReqLLM.Embedding.embed("openai:text-embedding-3-small", "Hello world")
      #=> {:ok, [0.1, -0.2, 0.3, ...]}

      # Multiple text inputs
      {:ok, embeddings} = ReqLLM.Embedding.embed(
        "openai:text-embedding-3-small",
        ["Hello", "World"]
      )
      #=> {:ok, [[0.1, -0.2, ...], [0.3, 0.4, ...]]}

      # Multimodal with ContentPart
      alias ReqLLM.Message.ContentPart

      {:ok, embedding} = ReqLLM.Embedding.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        ContentPart.video_url("https://example.com/video.mp4"),
        base_url: "http://localhost:8003"
      )

      # Batch multimodal
      {:ok, embeddings} = ReqLLM.Embedding.embed(
        "qwen_embed:Qwen3-VL-Embedding-8B",
        [
          ContentPart.video_url("https://example.com/video.mp4"),
          ContentPart.text("person sitting at desk")
        ]
      )

  """
  @spec embed(
          String.t() | {atom(), keyword()} | struct(),
          String.t() | ContentPart.t() | [String.t() | ContentPart.t()],
          keyword()
        ) :: {:ok, [float()] | [[float()]]} | {:error, term()}
  def embed(model_spec, input, opts \\ [])

  def embed(model_spec, text, opts) when is_binary(text) do
    with {:ok, model} <- validate_model(model_spec),
         :ok <- validate_input(text),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <- provider_module.prepare_request(:embedding, model, text, opts),
         {:ok, %Req.Response{status: status, body: decoded_response}} when status in 200..299 <-
           Req.request(request) do
      extract_single_embedding(decoded_response)
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: Request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  def embed(model_spec, texts, opts) when is_list(texts) do
    with {:ok, model} <- validate_model(model_spec),
         :ok <- validate_input(texts),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <- provider_module.prepare_request(:embedding, model, texts, opts),
         {:ok, %Req.Response{status: status, body: decoded_response}} when status in 200..299 <-
           Req.request(request) do
      extract_multiple_embeddings(decoded_response)
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: Request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  def embed(model_spec, %ContentPart{} = content_part, opts) do
    with {:ok, model} <- validate_model(model_spec),
         :ok <- validate_input(content_part),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <- provider_module.prepare_request(:embedding, model, content_part, opts),
         {:ok, %Req.Response{status: status, body: decoded_response}} when status in 200..299 <-
           Req.request(request) do
      extract_single_embedding(decoded_response)
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: Request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  defp validate_input("") do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "text: cannot be empty")}
  end

  defp validate_input(text) when is_binary(text) do
    :ok
  end

  defp validate_input([]) do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "texts: cannot be empty")}
  end

  defp validate_input(texts) when is_list(texts) do
    :ok
  end

  defp validate_input(%ContentPart{type: :text, text: ""}) do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "text: cannot be empty")}
  end

  defp validate_input(%ContentPart{type: :text, text: nil}) do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "text: cannot be nil")}
  end

  defp validate_input(%ContentPart{type: type, url: nil})
       when type in [:video_url, :image_url] do
    {:error, ReqLLM.Error.Invalid.Parameter.exception(parameter: "url: cannot be nil")}
  end

  defp validate_input(%ContentPart{type: type})
       when type in [:text, :video_url, :image_url, :image] do
    :ok
  end

  defp validate_input(%ContentPart{type: type}) do
    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter: "content_part type: #{inspect(type)} not supported for embeddings"
     )}
  end

  defp extract_single_embedding(%{"data" => [%{"embedding" => embedding}]}) do
    {:ok, embedding}
  end

  defp extract_single_embedding(response) do
    {:error,
     ReqLLM.Error.API.Response.exception(
       reason: "Invalid embedding response format",
       response_body: response
     )}
  end

  defp extract_multiple_embeddings(%{"data" => data}) when is_list(data) do
    embeddings =
      data
      |> Enum.sort_by(& &1["index"])
      |> Enum.map(& &1["embedding"])

    {:ok, embeddings}
  end

  defp extract_multiple_embeddings(response) do
    {:error,
     ReqLLM.Error.API.Response.exception(
       reason: "Invalid embedding response format",
       response_body: response
     )}
  end
end
