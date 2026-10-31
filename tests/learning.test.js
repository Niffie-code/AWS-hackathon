import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyProgress,saveWords,recordAttempt,wordStatus,reminderDue,loadProgress} from '../src/learning.js';
const word={word:'mitigate',definition:'Reduce harm',challenge:'Write a sentence',pronunciation:'MIT',nuance:'formal',spectrum_comparison:'reduce'};
test('saved words deduplicate and retain history across new sessions',()=>{
 let p=saveWords(emptyProgress(),[word],'IELTS');
 p=recordAttempt(p,word,'IELTS','First sentence',{score:60,feedback:'More context'});
 p=saveWords(p,[{...word,word:'MITIGATE'}],'Work');
 assert.equal(p.words.length,1);assert.equal(p.words[0].attempts.length,1);
 assert.equal(wordStatus(p.words[0]),'Needs practice');
 p=recordAttempt(p,word,'IELTS','Revised sentence',{score:94,feedback:'Precise'});
 assert.equal(wordStatus(p.words[0]),'Confident');
 assert.equal(p.words[0].attempts[1].score-p.words[0].attempts[0].score,34);
 assert.deepEqual(loadProgress({getItem:()=>JSON.stringify(p)}),p);
});
test('storage corruption and unavailability do not crash learning',()=>{
 assert.deepEqual(loadProgress({getItem:()=>'{broken'}),emptyProgress());
 assert.deepEqual(loadProgress({getItem:()=>{throw Error('denied')}}),emptyProgress());
 assert.deepEqual(loadProgress({getItem:()=>JSON.stringify({words:[{word:10}]})}),emptyProgress());
});
test('daily reminders respect enabled state, time, local date, and delivery marker',()=>{
 const now=new Date(2026,9,31,18,0);
 assert.equal(reminderDue({enabled:false,time:'18:00'},now,''),null);
 assert.equal(reminderDue({enabled:true,time:'19:00'},now,''),null);
 assert.equal(reminderDue({enabled:true,time:'18:00'},now,''),'2026-10-31');
 assert.equal(reminderDue({enabled:true,time:'18:00'},now,'2026-10-31'),null);
 assert.equal(reminderDue({enabled:true,time:'18:00'},new Date(2026,10,1,18,0),'2026-10-31'),'2026-11-1');
});
