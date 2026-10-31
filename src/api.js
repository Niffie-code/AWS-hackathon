import { mockVocabulary, mockEvaluation } from './mockData';
// Defaults to mock locally; set VITE_USE_MOCK=false for a deployed API.
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
const delay = () => new Promise(resolve => setTimeout(resolve, 1200 + Math.random() * 600));
async function post(path, body) {
  try {
    const response = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
    const data = await response.json();
    return response.ok ? data : {error: data.error || 'The coach could not complete your request.', ...(data.raw_text ? {raw_text:data.raw_text} : {})};
  } catch { return {error:'Could not reach the coach. Please try again.'}; }
}
export async function generateVocab(goal) {
  if (!USE_MOCK) return post('/api/generate', {user_goal:goal});
  await delay(); return mockVocabulary;
}
export async function evaluateSentence(goal, sentence, targetWords) {
  if (!USE_MOCK) return post('/api/evaluate', {user_goal:goal, user_input:sentence, target_words:targetWords});
  await delay(); return mockEvaluation(sentence, targetWords);
}
