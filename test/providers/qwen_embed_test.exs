defmodule ReqLLM.Providers.QwenEmbedTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Providers.QwenEmbed

  describe "provider registration" do
    test "qwen_embed provider is registered" do
      assert :qwen_embed in ReqLLM.Providers.list()
    end

    test "can resolve qwen_embed provider" do
      assert {:ok, QwenEmbed} = ReqLLM.Providers.get(:qwen_embed)
    end
  end

  describe "model resolution" do
    test "can create model from string spec" do
      assert {:ok, model} = ReqLLM.model("qwen_embed:Qwen3-VL-Embedding-8B")
      assert model.provider == :qwen_embed
      assert model.id == "Qwen3-VL-Embedding-8B"
    end

    test "validates embedding support" do
      assert {:ok, _model} = ReqLLM.Embedding.validate_model("qwen_embed:Qwen3-VL-Embedding-8B")
    end
  end

  describe "prepare_request/4" do
    setup do
      {:ok, model} = ReqLLM.model("qwen_embed:Qwen3-VL-Embedding-8B")
      {:ok, model: model}
    end

    test "prepares text embedding request", %{model: model} do
      assert {:ok, request} = QwenEmbed.prepare_request(:embedding, model, "hello world", [])

      assert request.url.path == "/embed"
      assert request.method == :post
    end

    test "prepares video embedding request with ContentPart", %{model: model} do
      video_part = ContentPart.video_url("https://example.com/video.mp4")

      assert {:ok, request} =
               QwenEmbed.prepare_request(:embedding, model, video_part, [])

      assert request.options[:inputs] == [%{video: "https://example.com/video.mp4"}]
    end

    test "prepares image embedding request with ContentPart", %{model: model} do
      image_part = ContentPart.image_url("https://example.com/image.jpg")

      assert {:ok, request} =
               QwenEmbed.prepare_request(:embedding, model, image_part, [])

      assert request.options[:inputs] == [%{image: "https://example.com/image.jpg"}]
    end

    test "prepares text embedding request with ContentPart", %{model: model} do
      text_part = ContentPart.text("person sitting at desk")

      assert {:ok, request} =
               QwenEmbed.prepare_request(:embedding, model, text_part, [])

      assert request.options[:inputs] == [%{text: "person sitting at desk"}]
    end

    test "prepares batch multimodal embedding request", %{model: model} do
      inputs = [
        ContentPart.video_url("https://example.com/video.mp4"),
        ContentPart.text("person sitting at desk"),
        ContentPart.image_url("https://example.com/image.jpg")
      ]

      assert {:ok, request} =
               QwenEmbed.prepare_request(:embedding, model, inputs, [])

      assert request.options[:inputs] == [
               %{video: "https://example.com/video.mp4"},
               %{text: "person sitting at desk"},
               %{image: "https://example.com/image.jpg"}
             ]
    end

    test "rejects non-embedding operations", %{model: model} do
      assert {:error, error} = QwenEmbed.prepare_request(:chat, model, "hello", [])
      assert Exception.message(error) =~ "not supported"
    end
  end

  describe "encode_body/1" do
    test "encodes inputs to JSON body" do
      request = %Req.Request{
        options: %{
          inputs: [%{text: "hello"}, %{video: "https://example.com/video.mp4"}]
        }
      }

      encoded = QwenEmbed.encode_body(request)

      assert encoded.body == ~s({"inputs":[{"text":"hello"},{"video":"https://example.com/video.mp4"}]})
    end
  end
end
