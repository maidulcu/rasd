document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);

    const btn = form.querySelector('button[type="submit"]');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        if (res.redirected) {
            window.location.href = res.url;
        } else if (res.ok) {
            window.location.href = '/dashboard';
        } else {
            alert('Analysis failed. Try again.');
            btn.textContent = 'Analyze';
            btn.disabled = false;
        }
    } catch (err) {
        alert('Error: ' + err.message);
        btn.textContent = 'Analyze';
        btn.disabled = false;
    }
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
