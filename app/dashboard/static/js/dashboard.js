document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const fileInput = form.querySelector('input[name="video_file"]');
    const urlInput = form.querySelector('input[name="video_url"]');

    if ((!fileInput.files || fileInput.files.length === 0) && !urlInput.value.trim()) {
        alert('Please select a video file or enter a video URL.');
        return;
    }

    const formData = new FormData(form);
    const btn = form.querySelector('button[type="submit"]');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        if (res.redirected) {
            window.location.href = res.url;
        } else {
            const text = await res.text();
            const match = text.match(/Error:<\/strong>\s*([^<]+)/);
            alert(match ? match[1] : 'Analysis failed. Try again.');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
    btn.textContent = 'Analyze';
    btn.disabled = false;
});

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('people-detected').textContent = data.total_people;
        document.getElementById('objects-count').textContent = data.total_objects;
        document.getElementById('fps-display').textContent = data.fps;
    } catch (e) {}
}

loadStats();
setInterval(loadStats, 5000);
