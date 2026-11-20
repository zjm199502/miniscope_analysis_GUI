// Offline renderer. The preview and all exports use the same SVG tree.
const ns = 'http://www.w3.org/2000/svg';
const colors = ['#137b78','#c56627','#66559a','#3873a0'];
const $ = id => document.getElementById(id);
function element(parent, name, attrs={}, value=null) {
  const el = document.createElementNS(ns,name);
  for (const [key,v] of Object.entries(attrs)) el.setAttribute(key,v);
  if (value!==null) el.textContent=value;
  parent.append(el); return el;
}
function extent(rows) {
  let a=Infinity,b=-Infinity;
  for(const row of rows) for(const v of row) if(Number.isFinite(v)){a=Math.min(a,v);b=Math.max(b,v)}
  return a===Infinity?[0,1]:[a,b];
}
function figure(charts, cols) {
  const rows=Math.ceil(charts.length/cols), pw=390, ph=286;
  const svg=document.createElementNS(ns,'svg');
  svg.setAttribute('xmlns',ns); svg.setAttribute('width',pw*cols); svg.setAttribute('height',ph*rows);
  svg.setAttribute('viewBox',[0,0,pw*cols,ph*rows].join(' '));
  svg.setAttribute('font-family','Microsoft YaHei, PingFang SC, sans-serif');
  element(svg,'rect',{width:'100%',height:'100%',fill:'white'});
  charts.forEach((c,k)=>{
    const g=element(svg,'g',{transform:'translate('+pw*(k%cols)+' '+ph*Math.floor(k/cols)+')'});
    const txt=(x,y,value,size=12,attrs={})=>element(g,'text',{x,y,'font-size':size,fill:'#203a42',...attrs},value);
    txt(12,23,charts.length>1?String.fromCharCode(65+k)+'   '+c.title:c.title,16,{'font-weight':650});
    const heat=c.kind==='heat', ny=c.data.length, nx=c.data[0].length;
    const H=154,W=heat&&!c.x?H*(c.aspect||1):288,w=Math.min(W,288),h=heat&&!c.x?w/(c.aspect||1):H;
    const hh=Math.min(H,h), left=46,top=40;
    const [lo,hi]=c.bounds||extent(c.data);
    const [xmin,xmax]=c.x?extent(c.kind==='path'?c.x:[c.x]):[0,nx-1];
    const px=x=>left+(x-xmin)/(xmax-xmin||1)*w;
    const py=y=>top+hh-(y-lo)/(hi-lo||1)*hh;
    if(heat){
      const canvas=document.createElement('canvas');canvas.width=nx;canvas.height=ny;
      const ctx=canvas.getContext('2d');
      c.data.forEach((row,j)=>row.forEach((v,i)=>{
        const u=Math.min(1,Math.max(0,(v-lo)/(hi-lo||1)));
        ctx.fillStyle=v==null?'#d7dadb':'hsl('+(245-195*u)+' 65% '+(24+42*u)+'%)';
        ctx.fillRect(i,j,1,1);
      }));
      element(g,'image',{x:left,y:top,width:w,height:hh,preserveAspectRatio:'none',href:canvas.toDataURL(),style:'image-rendering:pixelated'});
      if(c.marker)txt(left+c.marker[0]*w,top+c.marker[1]*hh,'★',20,{'text-anchor':'middle','dominant-baseline':'central',fill:'white',stroke:'#232e34','stroke-width':1,'paint-order':'stroke'});
      txt(12,251,'色阶 '+lo.toPrecision(3)+' — '+hi.toPrecision(3)+'     灰色 · 缺测');
    } else {
      c.data.forEach((ys,j)=>{
        const xs=c.kind==='path'?c.x[j]:c.x;
        let line='',start=true;
        ys.forEach((v,i)=>{
          if(!Number.isFinite(v)){start=true;return}
          const x=px(xs?xs[i]:i),y=py(v);
          line+=(start?' M':' L')+x+' '+y;
          if(start||c.points)element(g,'circle',{cx:x,cy:y,r:2.5,fill:colors[j%4]});
          start=false;
        });
        element(g,'path',{d:line,stroke:colors[j%4],fill:'none','stroke-width':2});
      });
      txt(2,top+7,hi.toPrecision(3));txt(2,top+hh,lo.toPrecision(3));
    }
    if(c.zero||c.band!=null){
      if(c.band)element(g,'rect',{x:px(Math.max(xmin,-c.band)),y:top,width:px(Math.min(xmax,c.band))-px(Math.max(xmin,-c.band)),height:hh,fill:'#555',opacity:.14});
      element(g,'line',{x1:px(0),x2:px(0),y1:top,y2:top+hh,stroke:'#374649','stroke-dasharray':'3 3'});
    }
    element(g,'rect',{x:left,y:top,width:w,height:hh,fill:'none',stroke:'#a8b7b9'});
    txt(left,top+hh+15,c.points?'跨天':Number(xmin.toPrecision(3)));
    txt(left+w,top+hh+15,c.points?'日内':Number(xmax.toPrecision(3)),12,{'text-anchor':'end'});
    txt(12,229,c.axes,11);
    if(!heat)(c.labels||[]).slice(0,4).forEach((s,i)=>txt(12+i*94,254,'● '+s,11,{fill:colors[i%4]}));
  });
  return svg;
}
function save(blob,name){
  const a=document.createElement('a'),url=URL.createObjectURL(blob);
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);
}
// Store uncompressed ZIP entries so figure, values and metadata travel together.
function zip(files){
  const enc=new TextEncoder(),local=[],central=[];let offset=0,size=0;
  function header(n,fields){const bytes=new Uint8Array(n),d=new DataView(bytes.buffer);for(const [p,v,w] of fields)d[w===4?'setUint32':'setUint16'](p,v,true);return bytes}
  for(const [name,data] of files){
    const fname=enc.encode(name);let crc=-1;
    for(const b of data){crc^=b;for(let k=0;k<8;k++)crc=(crc>>>1)^((crc&1)?0xedb88320:0)}
    crc=(~crc)>>>0;
    const lengths=[[14,crc,4],[18,data.length,4],[22,data.length,4],[26,fname.length]];
    const lh=header(30,[[0,0x04034b50,4],[4,20],[6,2048],...lengths]);
    const ch=header(46,[[0,0x02014b50,4],[4,20],[6,20],[8,2048],...lengths.map(([p,v,w])=>[p+2,v,w]),[42,offset,4]]);
    local.push(lh,fname,data);central.push(ch,fname);offset+=30+fname.length+data.length;size+=46+fname.length;
  }
  return new Blob([...local,...central,header(22,[[0,0x06054b50,4],[8,files.length],[10,files.length],[12,size,4],[16,offset,4]])],{type:'application/zip'});
}
async function download(charts,svg,name){
  const enc=new TextEncoder(),fmt=$('format').value;
  const copy=svg.cloneNode(true);
  const mm=Number($('width').value), vb=svg.viewBox.baseVal;
  copy.setAttribute('width',mm+'mm');copy.setAttribute('height',mm*vb.height/vb.width+'mm');
  const xml=new XMLSerializer().serializeToString(copy);let bytes=enc.encode(xml);
  if(fmt==='png'){
    const img=new Image(),url=URL.createObjectURL(new Blob([xml],{type:'image/svg+xml'}));
    try{img.src=url;await img.decode();const cv=document.createElement('canvas');cv.width=Math.round(mm/25.4*300);cv.height=Math.round(cv.width*vb.height/vb.width);cv.getContext('2d').drawImage(img,0,0,cv.width,cv.height);
    const blob=await new Promise(ok=>cv.toBlob(ok));bytes=new Uint8Array(await blob.arrayBuffer());}finally{URL.revokeObjectURL(url)}
  }
  const csv=[['panel','title','series_or_row','column','x','value']];
  charts.forEach((c,k)=>c.data.forEach((row,j)=>row.forEach((v,i)=>csv.push([k+1,c.title,c.labels?.[j]??j,i,(c.kind==='path'?c.x[j]:c.x)?.[i]??i,v??'']))));
  const quote=v=>'"'+String(v).replaceAll('"','""')+'"';
  const stem=name.replace(/[\\/:*?"<>|]/g,'_');
  const meta={study:r.title,run:r.run,parameters:r.params,group:sel.value,width_mm:$('width').value,charts:charts.map(({data,...settings})=>settings)};
  save(zip([[stem+'.'+(fmt==='pdf'?'svg':fmt),bytes],[stem+'.csv',enc.encode('\ufeff'+csv.map(row=>row.map(quote).join(',')).join('\r\n'))],[stem+'.json',enc.encode(JSON.stringify(meta,null,2))]]),stem+'.zip');
}
const sel=$('group'),main=$('plots');
function preview(charts,cols,title){
  const card=document.createElement('article'),svg=figure(charts,cols);
  card.append(svg);main.append(card);
  const btn=document.createElement('button');btn.textContent=charts.length>1?'导出整版':'导出当前图';
  btn.onclick=async()=>{try{if($('format').value==='pdf'){await download(charts,svg,title);card.classList.add('printing');try{window.print()}finally{card.classList.remove('printing')}return}await download(charts,svg,title)}catch(e){alert('导出失败：'+e.message)}};
  card.append(btn);
}
function show(){
  main.replaceChildren();document.documentElement.style.setProperty('--width',$('width').value+'mm');
  let charts=r.charts.filter(c=>(c.group||'总览')===sel.value);
  let combo=charts.filter(c=>c.panel);
  if(r.params.mode==='wall'&&charts[0].group)combo=[charts[0],r.charts[0]];
  if(combo.length>1)preview(combo,r.params.mode==='wall'?1:2,r.title+'-'+sel.value);
  charts.forEach(c=>preview([c],1,r.title+'-'+sel.value+'-'+c.title));
}
$('title').textContent=document.title=r.title;$('note').textContent=r.note;
for(const g of new Set(r.charts.map(c=>c.group||'总览')))sel.add(new Option(g,g));
sel.onchange=$('width').onchange=show;show();

