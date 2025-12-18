using LLama;
using LLama.Common;
using System;

public class LlamaRunner
{
    private readonly LLamaContext _ctx;
    private readonly StatelessExecutor _executor;

    public LlamaRunner(string modelPath, int contextSize = 2048, int threads = 8)
    {
        var weights = LLamaWeights.LoadFromFile(modelPath);
        var parameters = new LLamaContextParams
        {
            ContextSize = contextSize,
            Threads = threads
        };
        _ctx = new LLamaContext(weights, parameters);
        _executor = new StatelessExecutor(_ctx);
    }

    public string Generate(
        string prompt,
        int maxTokens = 120,
        float temperature = 0.4f,
        float repetitionPenalty = 1.1f)
    {
        var inferParams = new InferenceParams
        {
            MaxTokens = maxTokens,
            Temperature = temperature,
            RepeatPenalty = repetitionPenalty,
            PenalizeRepeatTokens = true
        };

        try
        {
            return _executor.Infer(prompt, inferParams);
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"LLM generation failed: {ex}");
            return string.Empty;
        }
    }
}
