// build.js — reads index.template.html, injects MAPBOX_TOKEN, writes index.html
const fs = require('fs');

const token = process.env.MAPBOX_TOKEN;
if (!token) {
  console.error('Error: MAPBOX_TOKEN environment variable is not set.');
  process.exit(1);
}

const src = fs.readFileSync('index.template.html', 'utf8');

if (!src.includes('YOUR_MAPBOX_TOKEN')) {
  console.error('Error: placeholder YOUR_MAPBOX_TOKEN not found in index.template.html.');
  process.exit(1);
}

const out = src.replace('YOUR_MAPBOX_TOKEN', token);
fs.writeFileSync('index.html', out, 'utf8');

console.log('index.html written with Mapbox token injected.');
