// Cloudflare owns the clock; GitHub still performs all Python inference.
const repo = 'tp-duarte/aemo-gridcast';
const workflow = 'hourly.yml';
const stateKey = 'state/scheduler.json';
export async function tick(env, scheduledTime, fetcher = fetch) {
  const now = new Date(scheduledTime);
  const hour = now.toISOString().slice(0,13);
  const save = async (status, extra = {}) => {
    const result = {status, checked_at:now.toISOString(), ...extra};
    await env.DATA.put(stateKey, JSON.stringify(result));
    console.log(JSON.stringify(result));
    return result;
  };
  if (!env.GITHUB_DISPATCH_TOKEN) return save('needs_github_token');
  const pointer = await env.DATA.get('public/current.json');
  const latest = pointer ? await pointer.json() : {};
  if (latest.forecast_issued_at?.slice(0,13) === hour)
    return save('current_hour_saved', {forecast_issued_at:latest.forecast_issued_at});
  const headers = {Authorization:`Bearer ${env.GITHUB_DISPATCH_TOKEN}`,
    Accept:'application/vnd.github+json','User-Agent':'gridcast-cloudflare-scheduler',
    'X-GitHub-Api-Version':'2026-03-10'};
  const base = `https://api.github.com/repos/${repo}/actions/workflows/${workflow}`;
  const runsResponse = await fetcher(`${base}/runs?per_page=10`, {headers,signal:AbortSignal.timeout(15000)});
  if (!runsResponse.ok) {
    await save('github_read_failed',{http_status:runsResponse.status});
    throw new Error(`GitHub run lookup failed: ${runsResponse.status}`);
  }
  const runs = (await runsResponse.json()).workflow_runs;
  if (!Array.isArray(runs)) throw new Error('Invalid GitHub workflow response');
  if (runs.some(r=>r.status !== 'completed' && now-Date.parse(r.created_at) < 20*60000))
    return save('forecast_job_pending');
  const dispatched = await fetcher(`${base}/dispatches`, {method:'POST',headers:{...headers,'Content-Type':'application/json'},
    body:JSON.stringify({ref:'main'}),signal:AbortSignal.timeout(15000)});
  if (!dispatched.ok) {
    await save('github_dispatch_failed',{http_status:dispatched.status});
    throw new Error(`GitHub dispatch failed: ${dispatched.status}`);
  }
  return save('dispatched',{last_forecast_issued_at:latest.forecast_issued_at||null});
}
export default {
  async scheduled(controller,env) { await tick(env,controller.scheduledTime); },
  async fetch(request,env) {
    if (request.method !== 'GET' || new URL(request.url).pathname !== '/health')
      return new Response('Not found',{status:404});
    const stored = await env.DATA.get(stateKey);
    const state = stored ? await stored.json() : {status:'awaiting_first_tick'};
    const configured = Boolean(env.GITHUB_DISPATCH_TOKEN);
    return Response.json({configured,...state},{status:configured?200:503,headers:{'Cache-Control':'no-store'}});
  }
};
