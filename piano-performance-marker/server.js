const express = require('express');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const PORT = process.env.PORT || 3000;

// Paths
const ROOT_DIR = __dirname;
const BUILD_DIR = path.join(ROOT_DIR, 'build');
const EVALUATIONS_DIR = process.env.EVALUATIONS_DIR || path.join(ROOT_DIR, 'evaluations');
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || '';

// Middleware
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true, limit: '5mb' }));

// Ensure base evaluations directory exists on boot
ensureDir(EVALUATIONS_DIR).then(() => {
  console.log(`[init] Ensured evaluations dir: ${EVALUATIONS_DIR}`);
}).catch((e) => {
  console.error('[init] Failed to ensure evaluations dir:', e);
});

// Health check
app.get('/healthz', (req, res) => {
  res.json({ ok: true, uptime: process.uptime(), evalDir: EVALUATIONS_DIR });
});

// List users that have saved evaluations
app.get('/api/users', async (_req, res) => {
  try {
    await ensureDir(EVALUATIONS_DIR);
    const entries = await fs.readdir(EVALUATIONS_DIR, { withFileTypes: true });
    const users = [];
    for (const ent of entries) {
      if (ent.isDirectory()) {
        const username = ent.name;
        const userDir = path.join(EVALUATIONS_DIR, username);
        let count = 0;
        try {
          const files = await fs.readdir(userDir);
          count = files.filter((f) => f.endsWith('.json')).length;
        } catch (_) {}
        users.push({ username, count });
      }
    }
    users.sort((a, b) => a.username.localeCompare(b.username));
    res.json({ ok: true, users });
  } catch (error) {
    console.error('Error listing users:', error);
    res.status(500).json({ ok: false, error: 'Failed to list users' });
  }
});

// Serve a saved evaluation file directly
app.get('/files/:username/:filename', async (req, res) => {
  try {
    const { username, filename } = req.params;
    const base = path.join(EVALUATIONS_DIR, username);
    const filePath = path.join(base, filename);
    // Prevent path traversal
    const resolvedBase = path.resolve(base) + path.sep;
    const resolvedTarget = path.resolve(filePath);
    if (!resolvedTarget.startsWith(resolvedBase)) {
      return res.status(400).json({ ok: false, error: 'Invalid path' });
    }
    await fs.access(resolvedTarget);
    return res.sendFile(resolvedTarget);
  } catch (error) {
    return res.status(404).json({ ok: false, error: 'File not found' });
  }
});

// Ensure evaluations dir exists lazily on write
async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (_) {}
}

// API endpoints (ported from src/setupProxy.js)
app.post('/api/save-evaluation', async (req, res) => {
  try {
    const { username, filename, data } = req.body;
    if (!username || !filename || !data) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const userDir = path.join(EVALUATIONS_DIR, username);
    await ensureDir(EVALUATIONS_DIR);
    await ensureDir(userDir);
    const filePath = path.join(userDir, filename);
    const payload = (typeof data === 'string') ? data : JSON.stringify(data, null, 2);
    await fs.writeFile(filePath, payload, 'utf8');
    console.log(`Saved evaluation: ${username}/${filename} -> ${filePath}`);
    res.json({ success: true, message: `Evaluation saved to ${username}/${filename}`, path: filePath });
  } catch (error) {
    console.error('Error saving evaluation:', error);
    res.status(500).json({ error: 'Failed to save evaluation', details: error.message });
  }
});

app.get('/api/evaluations/:username', async (req, res) => {
  try {
    const { username } = req.params;
    const { withData } = req.query;
    const userDir = path.join(EVALUATIONS_DIR, username);
    try {
      await fs.access(userDir);
    } catch {
      return res.json({ evaluations: [] });
    }
    const files = await fs.readdir(userDir);
    const jsonFiles = files.filter((f) => f.endsWith('.json'));
    const evaluations = await Promise.all(
      jsonFiles.map(async (file) => {
        const filePath = path.join(userDir, file);
        const stats = await fs.stat(filePath);
        const content = await fs.readFile(filePath, 'utf8');
        const data = JSON.parse(content);
        const base = {
          filename: file,
          timestamp: data.timestamp || stats.mtime,
          audioFile: data.audio_filename || data.audio_file,
          score: data.total_score,
          size: stats.size,
        };
        if (withData === '1' || withData === 'true') base.data = data;
        return base;
      })
    );
    evaluations.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    res.json({ evaluations });
  } catch (error) {
    console.error('Error fetching evaluations:', error);
    res.status(500).json({ error: 'Failed to fetch evaluations', details: error.message });
  }
});

app.get('/api/evaluation/:username/:filename', async (req, res) => {
  try {
    const { username, filename } = req.params;
    const filePath = path.join(EVALUATIONS_DIR, username, filename);
    const content = await fs.readFile(filePath, 'utf8');
    const data = JSON.parse(content);
    res.json({ ok: true, data });
  } catch (error) {
    console.error('Error reading evaluation file:', error);
    res.status(404).json({ ok: false, error: 'Evaluation not found' });
  }
});

app.get('/api/get-evaluation', async (req, res) => {
  try {
    const { username, filename } = req.query;
    if (!username || !filename) {
      return res.status(400).json({ ok: false, error: 'Missing username or filename' });
    }
    const filePath = path.join(EVALUATIONS_DIR, username, filename);
    const content = await fs.readFile(filePath, 'utf8');
    const data = JSON.parse(content);
    res.json({ ok: true, data });
  } catch (error) {
    // Not found → 200 with ok:false to avoid console red errors in UI
    if (error && error.code === 'ENOENT') {
      return res.json({ ok: false, error: 'Evaluation not found' });
    }
    console.error('Error reading evaluation file (query):', error);
    res.status(500).json({ ok: false, error: 'Failed to read evaluation' });
  }
});

// Admin helper
function requireAdmin(req, res) {
  if (!ADMIN_TOKEN) {
    res.status(501).json({ ok: false, error: 'ADMIN_TOKEN not configured' });
    return false;
  }
  const token = req.query.token || req.headers['x-admin-token'];
  if (token !== ADMIN_TOKEN) {
    res.status(403).json({ ok: false, error: 'Forbidden' });
    return false;
  }
  return true;
}

// Delete a specific evaluation file (admin only)
app.delete('/api/evaluation/:username/:filename', async (req, res) => {
  try {
    if (!requireAdmin(req, res)) return;
    const { username, filename } = req.params;
    const filePath = path.join(EVALUATIONS_DIR, username, filename);
    await fs.unlink(filePath);
    res.json({ ok: true, deleted: `${username}/${filename}` });
  } catch (error) {
    res.status(404).json({ ok: false, error: 'Delete failed or not found' });
  }
});

// Delete all evaluations for a user (admin only)
app.delete('/api/evaluations/:username', async (req, res) => {
  try {
    if (!requireAdmin(req, res)) return;
    const { username } = req.params;
    const userDir = path.join(EVALUATIONS_DIR, username);
    const files = await fs.readdir(userDir);
    let count = 0;
    for (const f of files) {
      if (f.endsWith('.json')) {
        try { await fs.unlink(path.join(userDir, f)); count++; } catch (_) {}
      }
    }
    res.json({ ok: true, username, deleted: count });
  } catch (error) {
    res.status(404).json({ ok: false, error: 'Delete failed' });
  }
});

app.get('/api/evaluation-latest', async (req, res) => {
  try {
    const { username, audio } = req.query;
    if (!username || !audio) {
      return res.status(400).json({ ok: false, error: 'Missing username or audio' });
    }
    const userDir = path.join(EVALUATIONS_DIR, username);
    try {
      await fs.access(userDir);
    } catch {
      return res.json({ ok: true, found: false });
    }
    const files = await fs.readdir(userDir);
    const jsonFiles = files.filter((f) => f.endsWith('.json'));
    const norm = (s) => String(s || '').toLowerCase();
    const base = norm(audio).replace(/\.[^.]+$/, '');
    const num = (str) => {
      const m = String(str).match(/(\d+)/);
      return m ? parseInt(m[1], 10) : null;
    };
    const candidates = [];
    for (const file of jsonFiles) {
      try {
        const filePath = path.join(userDir, file);
        const stats = await fs.stat(filePath);
        const content = await fs.readFile(filePath, 'utf8');
        const data = JSON.parse(content);
        const af = norm(data.audio_file);
        const stem = norm(file).replace(/\.[^.]+$/, '');
        const cond =
          af === norm(audio) ||
          af.endsWith('/' + norm(audio)) ||
          af.includes(base) ||
          stem.startsWith(base + '_') ||
          (num(af) && num(audio) && num(af) === num(audio));
        if (cond) {
          candidates.push({ filename: file, timestamp: data.timestamp || stats.mtime, data });
        }
      } catch (_) {}
    }
    if (candidates.length === 0) return res.json({ ok: true, found: false });
    candidates.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    const latest = candidates[0];
    return res.json({ ok: true, found: true, filename: latest.filename, data: latest.data });
  } catch (error) {
    console.error('Error evaluation-latest:', error);
    res.status(500).json({ ok: false, error: 'Failed to fetch latest evaluation' });
  }
});

app.get('/api/last-evaluation/:username', async (req, res) => {
  try {
    const { username } = req.params;
    const userDir = path.join(EVALUATIONS_DIR, username);
    try {
      await fs.access(userDir);
    } catch {
      return res.json({ exists: false, lastAudioIndex: null });
    }
    const files = await fs.readdir(userDir);
    const jsonFiles = files.filter((f) => f.endsWith('.json'));
    if (jsonFiles.length === 0) {
      return res.json({ exists: true, lastAudioIndex: null, totalEvaluations: 0 });
    }
    let maxAudioIndex = 0;
    for (const file of jsonFiles) {
      const name = file.replace(/\.json$/i, '');
      const m = name.match(/(\d+)/);
      if (m) {
        const n = parseInt(m[1], 10);
        if (!Number.isNaN(n) && n > maxAudioIndex) maxAudioIndex = n;
      } else {
        try {
          const filePath = path.join(userDir, file);
          const content = await fs.readFile(filePath, 'utf8');
          const data = JSON.parse(content);
          if (typeof data.audio_index === 'number' && data.audio_index > maxAudioIndex) {
            maxAudioIndex = data.audio_index;
          } else if (data.audio_filename || data.audio_file) {
            const s = String(data.audio_filename || data.audio_file);
            const mm = s.match(/(\d+)/);
            if (mm) {
              const nn = parseInt(mm[1], 10);
              if (!Number.isNaN(nn) && nn > maxAudioIndex) maxAudioIndex = nn;
            }
          }
        } catch (_) {}
      }
    }
    res.json({ exists: true, lastAudioIndex: maxAudioIndex || null, totalEvaluations: jsonFiles.length });
  } catch (error) {
    console.error('Error checking last evaluation:', error);
    res.status(500).json({ error: 'Failed to check last evaluation', details: error.message });
  }
});

// Serve static production build
app.use(express.static(BUILD_DIR));

// SPA fallback for non-API routes (Express 5 path-to-regexp compatible)
app.get(/^(?!\/api\/).*/, (req, res) => {
  res.sendFile(path.join(BUILD_DIR, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`piano-performance-marker running on :${PORT}`);
});
