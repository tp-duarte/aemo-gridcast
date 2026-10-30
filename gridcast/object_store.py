"""Small object-store interface; local integration tests need no cloud credentials."""
import os
from pathlib import Path

class LocalObjects:
 def __init__(self,root):self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
 def path(self,key):
  p=(self.root/key).resolve()
  if not p.is_relative_to(self.root):raise ValueError('Invalid object key')
  return p
 def get(self,key):return self.path(key).read_bytes()
 def put(self,key,body,immutable=False):
  p=self.path(key);p.parent.mkdir(parents=True,exist_ok=True)
  if immutable and p.exists():
   if p.read_bytes()!=body:raise ValueError('Immutable object conflict')
   return
  tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(body);tmp.replace(p)
 def keys(self,prefix):return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file() and p.relative_to(self.root).as_posix().startswith(prefix))
 def delete_public_cache(self,key):
  if not key.startswith('public/snapshots/'):raise ValueError('Only derived public cache may be pruned')
  self.path(key).unlink(missing_ok=True)

class R2Objects:
 def __init__(self):
  import boto3
  from botocore.config import Config
  self.bucket=os.environ['R2_BUCKET']
  self.client=boto3.client('s3',endpoint_url=os.environ['R2_ENDPOINT'],aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],region_name='auto',config=Config(retries={'max_attempts':4,'mode':'standard'},connect_timeout=10,read_timeout=60))
 def get(self,key):
  try:return self.client.get_object(Bucket=self.bucket,Key=key)['Body'].read()
  except self.client.exceptions.ClientError as exc:
   if exc.response['Error']['Code'] in ['NoSuchKey','404']:raise FileNotFoundError(key) from exc
   raise
 def put(self,key,body,immutable=False):
  # One GitHub concurrency group owns writes. No other writer may use this bucket prefix.
  if immutable:
   try:old=self.get(key)
   except FileNotFoundError:old=None
   if old is not None:
    if old!=body:raise ValueError('Immutable object conflict')
    return
  self.client.put_object(Bucket=self.bucket,Key=key,Body=body)
 def keys(self,prefix):
  return [o['Key'] for page in self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket,Prefix=prefix) for o in page.get('Contents',[])]
 def delete_public_cache(self,key):
  if not key.startswith('public/snapshots/'):raise ValueError('Only derived public cache may be pruned')
  self.client.delete_object(Bucket=self.bucket,Key=key)
