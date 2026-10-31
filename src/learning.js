export const STORAGE_KEY = 'lexigemma.learning.v1';
export const emptyProgress = () => ({words:[]});
export function saveWords(progress, words, goal) {
  const saved = [...progress.words];
  words.forEach(word => {
    const index = saved.findIndex(w => w.word.toLowerCase() === word.word.toLowerCase());
    if (index < 0) saved.push({...word, goal, attempts:[]});
  });
  return {words:saved};
}
export function recordAttempt(progress, word, goal, sentence, review) {
  const next = saveWords(progress,[word],goal);
  return {words:next.words.map(w => w.word.toLowerCase() === word.word.toLowerCase()
    ? {...w, attempts:[...w.attempts,{sentence,score:review.score,feedback:review.feedback,date:new Date().toISOString()}].slice(-100)} : w)};
}
export function wordStatus(word) {
  const score=word.attempts.at(-1)?.score;
  return score == null ? 'Not practiced' : score >= 90 ? 'Confident' : 'Needs practice';
}
export function loadProgress(storage) {
  try {
    const data=JSON.parse(storage.getItem(STORAGE_KEY));
    if(!Array.isArray(data?.words)) return emptyProgress();
    return {words:data.words.filter(w=>w && ['word','definition','challenge','pronunciation','nuance','spectrum_comparison','goal'].every(k=>typeof w[k]==='string') && Array.isArray(w.attempts)).map(w=>({...w,attempts:w.attempts.filter(a=>Number.isInteger(a.score)&&a.score>=0&&a.score<=100&&typeof a.sentence==='string'&&typeof a.feedback==='string')}))};
  } catch {return emptyProgress()}
}
export function reminderDue(settings, now, lastDate) {
  const day=`${now.getFullYear()}-${now.getMonth()+1}-${now.getDate()}`;
  const clock=`${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
  return settings.enabled && /^\d{2}:\d{2}$/.test(settings.time) && clock>=settings.time && day!==lastDate ? day : null;
}
