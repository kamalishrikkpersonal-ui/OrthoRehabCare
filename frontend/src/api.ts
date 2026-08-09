// API client for the OrthoRehab AI backend.
// All requests go to VITE_API_BASE_URL. No API keys live in the frontend.

import type { HealthResponse, Step5Result } from './types';

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') ??
  'http://localhost:8000';

export function getBaseUrl(): string {
  return BASE_URL;
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data && typeof data.detail === 'string') detail = data.detail;
      else if (data && typeof data.detail === 'object') detail = JSON.stringify(data.detail);
    } catch {
      // ignore parse failure
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/api/health`);
  return handleResponse<HealthResponse>(res);
}

export interface AnalyzePayload {
  video: File;
  referenceVideo?: File;
}

export async function analyzeExercise(
  payload: AnalyzePayload,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal
): Promise<Step5Result> {
  const form = new FormData();

  form.append('video', payload.video);

  if (payload.referenceVideo) {
    form.append('reference_video', payload.referenceVideo);
  }

  if (onProgress) {
    onProgress(10);
  }

  const res = await fetch(`${BASE_URL}/api/v1/exercises/analyze`, {
    method: 'POST',
    body: form,
    signal,
  });

  if (onProgress) {
    onProgress(95);
  }

  const result = await handleResponse<Step5Result>(res);

  if (onProgress) {
    onProgress(100);
  }

  return result;
}