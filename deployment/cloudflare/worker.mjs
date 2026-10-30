const regions=new Set(['NSW1','QLD1','VIC1','SA1','TAS1','NEM']);
export default {
 async fetch(request,env){
  const url=new URL(request.url),origin=request.headers.get('Origin');
  const cors={'Vary':'Origin','Access-Control-Allow-Methods':'GET, OPTIONS','Access-Control-Allow-Headers':'Content-Type'};
  if(origin){if(origin!==env.ALLOWED_ORIGIN)return new Response('Origin not allowed',{status:403});cors['Access-Control-Allow-Origin']=origin;}
  if(request.method==='OPTIONS')return new Response(null,{status:204,headers:cors});
  if(request.method!=='GET')return new Response('Read-only API',{status:405,headers:cors});
  if(!['/health','/v1/dashboard','/v1/forecasts.csv'].includes(url.pathname))return new Response('Not found',{status:404,headers:cors});
  const pointer=await env.DATA.get('public/current.json');
  if(!pointer)return Response.json({status:'awaiting_first_forecast'},{status:503,headers:cors});
  const meta=await pointer.json(),age=(Date.now()-Date.parse(meta.forecast_issued_at))/1000;
  if(!Number.isFinite(age)||age>5400||age< -60)return Response.json({status:'stale',forecast_issued_at:meta.forecast_issued_at,age_seconds:age},{status:503,headers:cors});
  if(url.pathname==='/health')return Response.json({status:'ok',...meta,age_seconds:age},{headers:{...cors,'Cache-Control':'no-store'}});
  const region=url.searchParams.get('region')||'NEM',mode=url.searchParams.get('mode')||'live';
  if(!regions.has(region)||mode!=='live')return Response.json({error:'This endpoint serves live saved forecasts only',available_modes:['live']},{status:400,headers:cors});
  if(!/^\d{8}T\d{12}$/.test(meta.snapshot))return new Response('Invalid snapshot pointer',{status:503,headers:cors});
  const csv=url.pathname.endsWith('.csv');const item=await env.DATA.get(`public/snapshots/${meta.snapshot}/${region}.${csv?'csv':'json'}`);
  if(!item)return new Response('Snapshot incomplete',{status:503,headers:cors});
  return new Response(item.body,{headers:{...cors,'Content-Type':csv?'text/csv; charset=utf-8':'application/json; charset=utf-8','Cache-Control':'public, max-age=60','ETag':item.httpEtag,...(csv?{'Content-Disposition':`attachment; filename="forecast-${region}.csv"`}:{})}});
 }
};
