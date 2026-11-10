import assert from 'node:assert/strict';
import worker from '../deployment/cloudflare/worker.mjs';
let issued=new Date().toISOString(),reads=[];
const env={ALLOWED_ORIGIN:'https://portfolio.example',DATA:{async get(key){reads.push(key);if(key==='public/current.json')return {async json(){return {snapshot:'20261029T120000123456',forecast_issued_at:issued,generated_at:issued}}};return {body:'{"schema_version":"1.0"}',httpEtag:'"test"'};}}};
let r=await worker.fetch(new Request('https://api.example/v1/dashboard?region=NEM&mode=live',{headers:{Origin:env.ALLOWED_ORIGIN}}),env);assert.equal(r.status,200);assert.equal(r.headers.get('Access-Control-Allow-Origin'),env.ALLOWED_ORIGIN);
r=await worker.fetch(new Request('https://api.example/models/current.json'),env);assert.equal(r.status,404);
r=await worker.fetch(new Request('https://api.example/v1/dashboard?region=../../models'),env);assert.equal(r.status,400);
r=await worker.fetch(new Request('https://api.example/v1/dashboard',{headers:{Origin:'https://other.example'}}),env);assert.equal(r.status,403);
env.ALLOWED_ORIGINS='https://tpduarte.com,https://www.tpduarte.com,https://portfolio.example';
for(const origin of ['https://tpduarte.com','https://www.tpduarte.com','https://portfolio.example']){
 r=await worker.fetch(new Request('https://api.example/v1/dashboard',{headers:{Origin:origin}}),env);assert.equal(r.status,200);assert.equal(r.headers.get('Access-Control-Allow-Origin'),origin);
}
r=await worker.fetch(new Request('https://api.example/v1/dashboard',{headers:{Origin:'https://tpduarte.com.attacker.example'}}),env);assert.equal(r.status,403);
r=await worker.fetch(new Request('https://api.example/v1/dashboard?mode=backtest'),env);assert.equal(r.status,400);
r=await worker.fetch(new Request('https://api.example/v1/forecasts.csv?region=NSW1'),env);assert.equal(r.status,200);assert.match(r.headers.get('Content-Type'),/text\/csv/);
issued=new Date(Date.now()-91*60*1000).toISOString();r=await worker.fetch(new Request('https://api.example/health'),env);assert.equal(r.status,503);
assert.equal((await r.json()).status,'update_delayed');
for(const minutes of [91,300,1500]){
 issued=new Date(Date.now()-minutes*60*1000).toISOString();
 for(const path of ['/v1/dashboard?region=NEM','/v1/forecasts.csv?region=QLD1']){
  r=await worker.fetch(new Request('https://api.example'+path),env);
  assert.equal(r.status,200);assert.equal(r.headers.get('X-Forecast-Status'),'update_delayed');
  assert.equal(await r.text(),'{"schema_version":"1.0"}');
 }
}
issued='invalid';r=await worker.fetch(new Request('https://api.example/v1/dashboard'),env);assert.equal(r.status,503);
issued=new Date(Date.now()+120000).toISOString();r=await worker.fetch(new Request('https://api.example/v1/dashboard'),env);assert.equal(r.status,503);
const empty={...env,DATA:{async get(){return null;}}};r=await worker.fetch(new Request('https://api.example/v1/dashboard'),empty);assert.equal(r.status,503);
assert(reads.every(k=>k.startsWith('public/')));console.log('Worker routing, CORS, private-key isolation, CSV and stale-forecast checks passed.');

const archiveReads=[];
const archiveEnv={...env,DATA:{async get(key){archiveReads.push(key);if(key==='public/archive/index.json')return {async json(){return {revision:'20261110T113845783284',dates:['2026-10-30']}}};return {body:'archive-content'};}}};
for(const suffix of ['dashboard?date=2026-10-30&region=QLD1','forecasts.csv?date=all&region=NEM']){
 r=await worker.fetch(new Request('https://api.example/v1/'+suffix),archiveEnv);assert.equal(r.status,200);assert.equal(await r.text(),'archive-content');assert.equal(r.headers.get('Cache-Control'),'no-store');
}
r=await worker.fetch(new Request('https://api.example/v1/archive'),archiveEnv);assert.deepEqual((await r.json()).dates,['2026-10-30']);
for(const query of ['date=2026-10-29','date=../../models','date=all&region=PRIVATE']){r=await worker.fetch(new Request('https://api.example/v1/dashboard?'+query),archiveEnv);assert.equal(r.status,404);}
assert(!archiveReads.includes('public/current.json'));assert(archiveReads.every(k=>k.startsWith('public/archive/')));console.log('Archive dates, CSV, invalid selection and independence from live pointer passed.');

