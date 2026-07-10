import { useState } from 'react';
import { api, type InferenceResult } from '../api';

/** One in-flight generation against POST /api/inference. */
export function useInference() {
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const submit = async (request: {
    prompt: string;
    max_tokens: number;
    enable_speculation?: boolean;
    enable_comparison?: boolean;
  }) => {
    setLoading(true);
    setError('');
    try {
      setResult(await api.inference(request));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return { submit, result, loading, error };
}
