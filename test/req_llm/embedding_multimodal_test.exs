defmodule ReqLLM.EmbeddingMultimodalTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Embedding

  describe "embed/3 with ContentPart validation" do
    test "rejects empty text ContentPart" do
      part = ContentPart.text("")

      result =
        Embedding.embed(
          "qwen_embed:Qwen3-VL-Embedding-8B",
          part,
          base_url: "http://localhost:9999"
        )

      assert {:error, error} = result
      assert Exception.message(error) =~ "cannot be empty"
    end

    test "rejects unsupported ContentPart types" do
      part = ContentPart.thinking("some thinking")

      result =
        Embedding.embed(
          "qwen_embed:Qwen3-VL-Embedding-8B",
          part,
          base_url: "http://localhost:9999"
        )

      assert {:error, error} = result
      assert Exception.message(error) =~ "not supported for embeddings"
    end
  end

  describe "model resolution for custom providers" do
    test "can resolve custom provider models" do
      assert {:ok, model} = ReqLLM.model("qwen_embed:AnyModelName")
      assert model.provider == :qwen_embed
      assert model.id == "AnyModelName"
    end

    test "model has embedding capabilities" do
      {:ok, model} = ReqLLM.model("qwen_embed:TestModel")
      assert model.capabilities.embeddings == true
    end

    test "unknown providers return error" do
      assert {:error, :unknown_provider} = ReqLLM.model("unknown_provider:model")
    end
  end

  describe "supported_models/0" do
    test "returns list of embedding models" do
      models = Embedding.supported_models()
      assert is_list(models)
    end
  end
end
