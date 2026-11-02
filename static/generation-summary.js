let resultSummarySequence=0;
async function loadGenerationSummary(jobId){
    const sequence=++resultSummarySequence;
    const card=document.getElementById('generationSummary');
    if(!card)return;
    card.hidden=false;
    const title=document.getElementById('generationSummaryTitle');
    const description=document.getElementById('generationSummaryDescription');
    const notes=document.getElementById('generationSummaryNotes');
    title.textContent='Loading result details…';description.textContent='';notes.replaceChildren();
    card.dataset.state='pending';
    try{
        const response=await fetch(`/api/scan/report/${encodeURIComponent(jobId)}`);
        if(sequence!==resultSummarySequence)return;
        if(!response.ok)throw new Error('Saved result details are unavailable.');
        const report=await response.json();
        if(sequence!==resultSummarySequence)return;
        card.dataset.state=report.degraded||report.experimental||!report.mesh_passed||report.warnings.length?'review':'ready';
        title.textContent=report.experimental?'Experimental result':report.mesh_passed?'Mesh checked':'Mesh needs review';
        const used=report.engine_label||report.engine_used||'Unknown engine';
        description.textContent=`Generated with ${used}.`;
        if(report.degraded)description.textContent+=` Requested ${report.engine_requested||'another engine'}; a fallback produced this result.`;
        const messages=[];
        if(report.appearance_only)messages.push('Photo detail changes appearance; it does not reconstruct missing 3D features.');
        if(report.relative_scale)messages.push('Size is relative. Physical dimensions have not been measured.');
        if(!report.fine_geometry_verified)messages.push('Fine detail accuracy has not been verified against a real 3D reference.');
        messages.push(...report.warnings);
        for(const text of [...new Set(messages)]){
            const item=document.createElement('li');item.textContent=text;notes.appendChild(item);
        }
    }catch(error){
        if(sequence!==resultSummarySequence)return;
        card.dataset.state='unknown';title.textContent='Result details unavailable';
        description.textContent='You can inspect the model, but its generation history could not be checked.';
    }
}
