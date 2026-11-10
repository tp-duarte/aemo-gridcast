const regions=new Set(['NSW1','QLD1','VIC1','SA1','TAS1','NEM']);
export default {
 async fetch(request,env){
  const url=new URL(request.url),origin=request.headers.get('Origin');
  const cors={'Vary':'Origin','Access-Control-Allow-Methods':'GET, OPTIONS','Access-Control-Allow-Headers':'Content-Type'};
  const allowedOrigins=(env.ALLOWED_ORIGINS||env.ALLOWED_ORIGIN||'').split(',').map(value=>value.trim()).filter(Boolean);
  if(origin){if(!allowedOrigins.includes(origin))return new Response('Origin not allowed',{status:403});cors['Access-Control-Allow-Origin']=origin;}
  if(request.method==='OPTIONS')return new Response(null,{status:204,headers:cors});
  if(request.method!=='GET')return new Response('Read-only API',{status:405,headers:cors});
  if(!['/health','/v1/dashboard','/v1/forecasts.csv','/v1/archive'].includes(url.pathname))return new Response('Not found',{status:404,headers:cors});
  const archiveDate = url.searchParams.get('date');
  if (url.pathname === '/v1/archive' || archiveDate) {
   const indexObject = await env.DATA.get('public/archive/index.json');
   if (!indexObject) return Response.json({error:'Archive is being prepared'}, {status:503,headers:cors});
   const index = await indexObject.json();
   if (url.pathname === '/v1/archive') return Response.json(index,{headers:{...cors,'Cache-Control':'no-store'}});
   const region = url.searchParams.get('region') || 'NEM';
   if (!regions.has(region) || !(archiveDate === 'all' || index.dates.includes(archiveDate)))
    return Response.json({error:'Unknown archive date or region'},{status:404,headers:cors});
   if (!/^\d{8}T\d{12}$/.test(index.revision)) return new Response('Invalid archive pointer',{status:503,headers:cors});
   const csv = url.pathname.endsWith('.csv');
   const item = await env.DATA.get(`public/archive/${index.revision}/${archiveDate}/${region}.${csv?'csv':'json'}`);
   if (!item) return Response.json({error:'Archive unavailable'},{status:503,headers:cors});
   return new Response(item.body,{headers:{...cors,'Content-Type':csv?'text/csv; charset=utf-8':'application/json; charset=utf-8','Cache-Control':'no-store',...(csv?{'Content-Disposition':`attachment; filename="forecast-archive-${archiveDate}-${region}.csv"`}:{})}});
  }
  const pointer=await env.DATA.get('public/current.json');
  if(!pointer)return Response.json({status:'awaiting_first_forecast'},{status:503,headers:cors});
  const meta=await pointer.json(),age=(Date.now()-Date.parse(meta.forecast_issued_at))/1000;
  if(!Number.isFinite(age)||age< -60)return Response.json({status:'invalid_forecast_timestamp'},{status:503,headers:{...cors,'Cache-Control':'no-store'}});
  const delayed=age>5400;
  // Alert monitoring without withholding the immutable last known forecast.
  if(url.pathname==='/health')return Response.json({...meta,status:delayed?'update_delayed':'ok',age_seconds:age},{status:delayed?503:200,headers:{...cors,'Cache-Control':'no-store'}});
  cors['X-Forecast-Status']=delayed?'update_delayed':'current';
  cors['X-Forecast-Issued-At']=meta.forecast_issued_at;
  cors['Access-Control-Expose-Headers']='X-Forecast-Status, X-Forecast-Issued-At';
  const region=url.searchParams.get('region')||'NEM',mode=url.searchParams.get('mode')||'live';
  if(!regions.has(region)||mode!=='live')return Response.json({error:'This endpoint serves live saved forecasts only',available_modes:['live']},{status:400,headers:cors});
  if(!/^\d{8}T\d{12}$/.test(meta.snapshot))return new Response('Invalid snapshot pointer',{status:503,headers:cors});
  const csv=url.pathname.endsWith('.csv');const item=await env.DATA.get(`public/snapshots/${meta.snapshot}/${region}.${csv?'csv':'json'}`);
  if(!item)return new Response('Snapshot incomplete',{status:503,headers:cors});
  return new Response(item.body,{headers:{...cors,'Content-Type':csv?'text/csv; charset=utf-8':'application/json; charset=utf-8','Cache-Control':'public, max-age=60','ETag':item.httpEtag,...(csv?{'Content-Disposition':`attachment; filename="forecast-${region}.csv"`}:{})}});
 }
};

