const { createProxyMiddleware } = require('http-proxy-middleware');
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

module.exports = function(app) {
  // 評価データの保存先ディレクトリ
  const EVALUATIONS_DIR = path.join(__dirname, '..', 'evaluations');

  // Express ミドルウェアの設定
  app.use(express.json());

  // 評価データを保存するAPIエンドポイント
  app.post('/api/save-evaluation', async (req, res) => {
    try {
      const { username, filename, data } = req.body;
      
      if (!username || !filename || !data) {
        return res.status(400).json({ error: 'Missing required fields' });
      }

      // ユーザーディレクトリのパス
      const userDir = path.join(EVALUATIONS_DIR, username);
      
      // ディレクトリが存在しない場合は作成
      await fs.mkdir(userDir, { recursive: true });
      
      // ファイルパス
      const filePath = path.join(userDir, filename);
      
      // JSONデータを保存
      await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf8');
      
      console.log(`Saved evaluation: ${username}/${filename}`);
      res.json({ 
        success: true, 
        message: `Evaluation saved to ${username}/${filename}`,
        path: filePath
      });
      
    } catch (error) {
      console.error('Error saving evaluation:', error);
      res.status(500).json({ 
        error: 'Failed to save evaluation',
        details: error.message 
      });
    }
  });

  // ユーザーの評価データ一覧を取得
  app.get('/api/evaluations/:username', async (req, res) => {
    try {
      const { username } = req.params;
      const { withData } = req.query;
      const userDir = path.join(EVALUATIONS_DIR, username);
      
      // ディレクトリが存在しない場合
      try {
        await fs.access(userDir);
      } catch {
        return res.json({ evaluations: [] });
      }
      
      // ファイル一覧を取得
      const files = await fs.readdir(userDir);
      const jsonFiles = files.filter(file => file.endsWith('.json'));
      
      // 各ファイルの情報を取得
      const evaluations = await Promise.all(
        jsonFiles.map(async (file) => {
          const filePath = path.join(userDir, file);
          const stats = await fs.stat(filePath);
          const content = await fs.readFile(filePath, 'utf8');
          const data = JSON.parse(content);
          const base = {
            filename: file,
            timestamp: data.timestamp || stats.mtime,
            audioFile: (data.audio_filename || data.audio_file),
            score: data.total_score,
            size: stats.size
          };
          if (withData === '1' || withData === 'true') {
            base.data = data;
          }
          return base;
        })
      );
      
      // タイムスタンプでソート（新しい順）
      evaluations.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      res.json({ evaluations });
      
    } catch (error) {
      console.error('Error fetching evaluations:', error);
      res.status(500).json({ 
        error: 'Failed to fetch evaluations',
        details: error.message 
      });
    }
  });

  // 特定の評価データを取得
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

  // クエリ版（ドットやエンコードの問題を避けるため）
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
      console.error('Error reading evaluation file (query):', error);
      res.status(404).json({ ok: false, error: 'Evaluation not found' });
    }
  });

  // 指定ユーザー＋音声に対する最新評価データを返却
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
      const jsonFiles = files.filter(file => file.endsWith('.json'));

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
          const cond = af === norm(audio) || af.endsWith('/' + norm(audio)) ||
            af.includes(base) || stem.startsWith(base + '_') ||
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

  // 特定ユーザーの最後に評価した音声番号を取得
  app.get('/api/last-evaluation/:username', async (req, res) => {
    try {
      const { username } = req.params;
      const userDir = path.join(EVALUATIONS_DIR, username);
      
      // ディレクトリが存在しない場合
      try {
        await fs.access(userDir);
      } catch {
        return res.json({ exists: false, lastAudioIndex: null });
      }
      
      // ファイル一覧を取得
      const files = await fs.readdir(userDir);
      const jsonFiles = files.filter(file => file.endsWith('.json'));

      if (jsonFiles.length === 0) {
        return res.json({ exists: true, lastAudioIndex: null, totalEvaluations: 0 });
      }

      // 保存形式は <basename>.json（例: amateur_piano_3.json）
      // 最後に完了した番号 = ファイル名から抽出した数値の最大値
      let maxAudioIndex = 0;
      for (const file of jsonFiles) {
        const name = file.replace(/\.json$/i, '');
        const m = name.match(/(\d+)/);
        if (m) {
          const n = parseInt(m[1], 10);
          if (!Number.isNaN(n) && n > maxAudioIndex) maxAudioIndex = n;
        } else {
          // 後方互換: 内容に audio_index や audio_filename がある場合も考慮
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

      res.json({ 
        exists: true,
        lastAudioIndex: maxAudioIndex || null,
        totalEvaluations: jsonFiles.length
      });
      
    } catch (error) {
      console.error('Error checking last evaluation:', error);
      res.status(500).json({ 
        error: 'Failed to check last evaluation',
        details: error.message 
      });
    }
  });

  console.log('API endpoints configured on port 5556');
};
