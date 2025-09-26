import React, { useState, useRef, useCallback, useEffect } from 'react';
import LoginScreen from './components/LoginScreen';
import AudioWaveform from './components/AudioWaveform';
import AudioControls from './components/AudioControls';
import MarkingsList from './components/MarkingsList';
import ScoreSlider from './components/ScoreSlider';
// import EvaluationHistory from './components/EvaluationHistory';
import { saveToUserDirectory } from './utils/dataExporter';
import './styles/waveform.css';

function App() {
  // 事前定義された音声ファイルリスト（長い順にソート）
  const AUDIO_FILES = [
    { url: '/audio/amateur_piano_7.wav', name: 'Amateur Piano Performance 7 (Longest)' },
    { url: '/audio/amateur_piano_6.wav', name: 'Amateur Piano Performance 6' },
    { url: '/audio/amateur_piano_5.wav', name: 'Amateur Piano Performance 5' },
    { url: '/audio/amateur_piano_4.wav', name: 'Amateur Piano Performance 4' },
    { url: '/audio/amateur_piano_10.wav', name: 'Amateur Piano Performance 10' },
    { url: '/audio/amateur_piano_3.wav', name: 'Amateur Piano Performance 3' },
    { url: '/audio/amateur_piano_9.wav', name: 'Amateur Piano Performance 9' },
    { url: '/audio/amateur_piano_8.wav', name: 'Amateur Piano Performance 8' },
    { url: '/audio/amateur_piano_2.wav', name: 'Amateur Piano Performance 2' },
    { url: '/audio/amateur_piano_1.wav', name: 'Amateur Piano Performance 1 (Shortest)' },
    // Extra 10 files (keep existing 1..10 URLs intact; add 11..20)
    { url: '/audio/amateur_piano_11.wav', name: 'Amateur Piano Performance 11' },
    { url: '/audio/amateur_piano_12.wav', name: 'Amateur Piano Performance 12' },
    { url: '/audio/amateur_piano_13.wav', name: 'Amateur Piano Performance 13' },
    { url: '/audio/amateur_piano_14.wav', name: 'Amateur Piano Performance 14' },
    { url: '/audio/amateur_piano_15.wav', name: 'Amateur Piano Performance 15' },
    { url: '/audio/amateur_piano_16.wav', name: 'Amateur Piano Performance 16' },
    { url: '/audio/amateur_piano_17.wav', name: 'Amateur Piano Performance 17' },
    { url: '/audio/amateur_piano_18.wav', name: 'Amateur Piano Performance 18' },
    { url: '/audio/amateur_piano_19.wav', name: 'Amateur Piano Performance 19' },
    { url: '/audio/amateur_piano_20.wav', name: 'Amateur Piano Performance 20' }
  ];

  // ファイル名（数値を考慮）で昇順に並べ替え
  AUDIO_FILES.sort((a, b) => {
    const fa = a.url.split('/').pop();
    const fb = b.url.split('/').pop();
    return fa.localeCompare(fb, undefined, { numeric: true, sensitivity: 'base' });
  });

  // 状態管理
  const [currentUser, setCurrentUser] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioMetadata, setAudioMetadata] = useState({
    name: '',
    duration: 0,
    sampleRate: 0
  });
  const [markings, setMarkings] = useState([]);
  const [score, setScore] = useState(5.0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [currentAudioIndex, setCurrentAudioIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [audioReady, setAudioReady] = useState(false);
  const [preloadedAudio, setPreloadedAudio] = useState(null);
  const [playingSection, setPlayingSection] = useState(null); // Track which section is playing
  
  const waveformRef = useRef(null);
  const lastAutoLoadedRef = useRef(null);

  // 候補評価が現在の音声に対応するかを厳密に判定
  const matchAudio = useCallback((candidate, baseFilename, displayName) => {
    if (!candidate) return false;
    const c = String(candidate).toLowerCase();
    const cbase = c.split('/').pop();
    const base = String(baseFilename || '').toLowerCase();
    const disp = String(displayName || '').toLowerCase();
    // 厳密: ファイル名一致 or 表示名の完全一致のみ許容
    if (cbase === base) return true;
    if (c === disp) return true;
    return false;
  }, []);

  // JSON取得ユーティリティ（HTML誤配信時の診断用）
  const fetchJson = useCallback(async (url) => {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const ct = res.headers.get('content-type') || '';
    const text = await res.text();
    if (!ct.includes('application/json')) {
      throw new Error(`Non-JSON response from ${url}: ` + text.slice(0, 200));
    }
    return JSON.parse(text);
  }, []);

  // 指定音声の既存評価を自動読み込み（<base>.json が存在する場合のみ）
  const tryLoadExistingEvaluation = useCallback(async (username, audioUrl) => {
    if (!username || !audioUrl) return false;
    try {
      const filename = audioUrl.split('/').pop() || '';
      const base = filename.replace(/\.[^.]+$/, '');
      const target = `${base}.json`;
      const j = await fetchJson(`/api/get-evaluation?username=${encodeURIComponent(username)}&filename=${encodeURIComponent(target)}`);
      if (!j.ok || !j.data) return false;
      const data = j.data;
      const loadedMarks = (data.problem_sections || []).map((s, i) => ({
        id: `mark_${Date.now()}_${i}`,
        start: parseFloat(Number(s.start).toFixed(1)),
        end: parseFloat(Number(s.end).toFixed(1)),
        duration: parseFloat((Number(s.end) - Number(s.start)).toFixed(1))
      }));
      setMarkings(loadedMarks);
      if (typeof data.total_score === 'number') setScore(data.total_score);
      console.log('[auto-load] applied evaluation (direct file):', target);
      return true;
    } catch (e) {
      return false;
    }
  }, [fetchJson]);

  // 音声URL解決（評価データから）
  const resolveAudioUrl = useCallback((audioName) => {
    if (!audioName) return null;
    if (/\.(wav|mp3|m4a|ogg)$/i.test(audioName)) {
      return `/audio/${audioName}`;
    }
    const found = AUDIO_FILES.find(a => a.name === audioName);
    if (found) return found.url;
    const m = String(audioName).match(/(\d+)/);
    if (m) {
      const idx = parseInt(m[1], 10);
      if (!isNaN(idx) && idx >= 1 && idx <= AUDIO_FILES.length) {
        return AUDIO_FILES[idx - 1].url;
      }
    }
    return null;
  }, []);

  const handleLoadEvaluation = useCallback((data, filename) => {
    try {
      const url = resolveAudioUrl(data.audio_file);
      if (!url) {
        alert('対応する音声ファイルが見つかりませんでした');
        return;
      }
      setIsPlaying(false);
      setCurrentTime(0);
      setAudioFile(url);
      setAudioReady(false);
      setIsLoading(true);

      const loadedMarks = (data.problem_sections || []).map((s, i) => ({
        id: `mark_${Date.now()}_${i}`,
        start: parseFloat(Number(s.start).toFixed(1)),
        end: parseFloat(Number(s.end).toFixed(1)),
        duration: parseFloat((Number(s.end) - Number(s.start)).toFixed(1))
      }));
      setMarkings(loadedMarks);
      if (typeof data.total_score === 'number') setScore(data.total_score);

      const idx = AUDIO_FILES.findIndex(a => a.url === url);
      if (idx >= 0) {
        setCurrentAudioIndex(idx);
        if (currentUser) localStorage.setItem(`${currentUser}_lastAudioIndex`, String(idx));
      }
    } catch (e) {
      console.error('Failed to load evaluation:', e);
      alert('評価データの読み込みでエラーが発生しました');
    }
  }, [AUDIO_FILES, currentUser, resolveAudioUrl]);

  // audioFile or currentUser が変わったら、その音声に紐づく最新評価を自動で読み込み
  useEffect(() => {
    const run = async () => {
      if (!currentUser || !audioFile) return;
      if (lastAutoLoadedRef.current === `${currentUser}|${audioFile}`) return;
      try {
        await tryLoadExistingEvaluation(currentUser, audioFile);
        lastAutoLoadedRef.current = `${currentUser}|${audioFile}`;
      } catch (e) {
        console.warn('自動読み込みに失敗:', e);
      }
    };
    run();
  }, [currentUser, audioFile, tryLoadExistingEvaluation]);

  // ログイン後に音声ファイルをロード（再開時は続きから）
  useEffect(() => {
    if (currentUser && AUDIO_FILES.length > 0) {
      // 再開位置が保存されていればそこから、なければ最初から
      const savedPosition = localStorage.getItem(`${currentUser}_lastAudioIndex`);
      const startIndex = savedPosition ? parseInt(savedPosition) : 0;
      console.log('Loading audio file at index:', startIndex);
      loadAudioFile(Math.min(startIndex, AUDIO_FILES.length - 1));
    }
  }, [currentUser]);

  // ログイン処理
  const handleLogin = (username, lastAudioIndex) => {
    setCurrentUser(username);
    // localStorageに保存（セッション維持用）
    localStorage.setItem('currentUser', username);
    
    // 再開位置を保存（未保存ならリセット）
    if (lastAudioIndex !== null && lastAudioIndex < AUDIO_FILES.length) {
      localStorage.setItem(`${username}_lastAudioIndex`, String(lastAudioIndex));
    } else {
      localStorage.removeItem(`${username}_lastAudioIndex`);
    }
  };

  // ログアウト処理
  const handleLogout = () => {
    if (currentUser) {
      // 現在の位置を保存
      localStorage.setItem(`${currentUser}_lastAudioIndex`, String(currentAudioIndex));
    }
    setCurrentUser(null);
    localStorage.removeItem('currentUser');
    setAudioFile(null);
    setCurrentAudioIndex(0);
    setAudioReady(false);
  };

  const loadAudioFile = async (index) => {
    if (index >= 0 && index < AUDIO_FILES.length) {
      console.log('loadAudioFile called with index:', index);
      setIsLoading(true);
      setAudioReady(false);
      const file = AUDIO_FILES[index];
      console.log('Loading file:', file.url, file.name);
      // 新しい音声へ切り替える直前に一旦クリア（前の音声の痕跡を残さない）
      setIsPlaying(false);
      setCurrentTime(0);
      setMarkings([]);
      setScore(5.0);
      setAudioFile(file.url);
      setAudioMetadata({
        name: file.name,
        duration: 0,
        sampleRate: 0
      });
      setCurrentAudioIndex(index);
      
      // 次の音声をプリロード
      if (index < AUDIO_FILES.length - 1) {
        const nextAudio = new Audio(AUDIO_FILES[index + 1].url);
        nextAudio.preload = 'auto';
        setPreloadedAudio(nextAudio);
      }

      // 既存の評価があれば自動読み込み（見つかった場合のみ上書き）
      if (currentUser) {
        await tryLoadExistingEvaluation(currentUser, file.url);
      }
    }
  };

  // マーキング追加
  const addMarking = useCallback((start, end) => {
    const newMarking = {
      id: `mark_${Date.now()}`,
      start: parseFloat(start.toFixed(1)),
      end: parseFloat(end.toFixed(1)),
      duration: parseFloat((end - start).toFixed(1))
    };
    
    setMarkings(prev => {
      // 重複チェック
      const hasOverlap = prev.some(m => 
        (start >= m.start && start <= m.end) ||
        (end >= m.start && end <= m.end) ||
        (start <= m.start && end >= m.end)
      );
      
      if (hasOverlap) {
        console.warn('Overlapping marking detected');
        return prev;
      }
      
      return [...prev, newMarking].sort((a, b) => a.start - b.start);
    });
  }, []);

  // マーキング削除
  const removeMarking = useCallback((id) => {
    setMarkings(prev => prev.filter(m => m.id !== id));
  }, []);

  // マーキング編集
  const updateMarking = useCallback((id, newStart, newEnd) => {
    setMarkings(prev => prev.map(m => 
      m.id === id 
        ? { ...m, start: newStart, end: newEnd, duration: newEnd - newStart }
        : m
    ).sort((a, b) => a.start - b.start));
  }, []);

  // データ保存（ユーザーディレクトリに自動保存）
  const handleSave = useCallback(() => {
    if (!currentUser) return;
    
    const audioFilename = (audioFile ? audioFile.split('/').pop() : '') || (audioMetadata.name || '').split('/').pop();
    const evaluationData = {
      evaluation_id: `eval_${Date.now()}`,
      evaluator: currentUser,
      // Canonical fields for perfect matching
      audio_filename: audioFilename,
      audio_display: audioMetadata.name,
      // Backward compatibility (keep existing field name as filename)
      audio_file: audioFilename,
      audio_index: currentAudioIndex + 1,
      duration: audioMetadata.duration,
      total_score: score,
      problem_sections: markings.map(m => ({
        start: m.start,
        end: m.end
      })),
      evaluation_time: Date.now(),
      timestamp: new Date().toISOString()
    };
    
    // 固定ファイル名（ユーザー配下で <音声ベース名>.json に上書き保存）
    const base = (audioFilename || `audio_${currentAudioIndex + 1}`)
      .replace(/\.[^/.]+$/, '')
      .replace(/[^a-zA-Z0-9_-]+/g, '_');
    const fileName = `${base}.json`;
    saveToUserDirectory(evaluationData, fileName, currentUser);
  }, [audioMetadata, score, markings, currentUser, currentAudioIndex]);

  // 次の音声へ
  const handleNext = useCallback(() => {
    handleSave();
    if (currentAudioIndex < AUDIO_FILES.length - 1) {
      const nextIndex = currentAudioIndex + 1;
      loadAudioFile(nextIndex);
      // 進捗を保存
      if (currentUser) {
        localStorage.setItem(`${currentUser}_lastAudioIndex`, String(nextIndex));
      }
    } else {
      alert('すべての音声の評価が完了しました');
      // 最初に戻る
      loadAudioFile(0);
      if (currentUser) {
        localStorage.setItem(`${currentUser}_lastAudioIndex`, '0');
      }
    }
  }, [currentAudioIndex, handleSave, currentUser]);

  // 前の音声へ
  const handlePrevious = useCallback(() => {
    if (currentAudioIndex > 0) {
      loadAudioFile(currentAudioIndex - 1);
    }
  }, [currentAudioIndex]);

  // キーボードショートカット
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (!audioFile) return;
      
      // スペースキーで再生/停止
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        setIsPlaying(prev => !prev);
      }
      // Enterキーで次へ
      else if (e.key === 'Enter') {
        e.preventDefault();
        handleNext();
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleNext, audioFile]);

  // セクション再生の終了でハイライト解除
  useEffect(() => {
    if (!isPlaying) setPlayingSection(null);
  }, [isPlaying]);

  // ログイン画面を表示
  if (!currentUser) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <h1 className="text-xl font-semibold text-gray-800">
                Piano Performance Marker
              </h1>
              <span className="text-sm bg-blue-100 text-blue-700 px-2 py-1 rounded">
                {currentUser}
              </span>
            </div>
            {audioMetadata.name && (
              <span className="text-sm text-gray-600">
                {audioMetadata.name} [{currentAudioIndex + 1}/{AUDIO_FILES.length}]
              </span>
            )}
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">
              Score: {score.toFixed(1)}/10
            </span>
            <button
              onClick={handleLogout}
              className="text-sm text-gray-500 hover:text-gray-700 underline"
            >
              ログアウト
            </button>
          </div>
        </div>
      </header>

      {/* メインコンテンツ */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {!audioFile ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-lg text-gray-600">音声ファイルの読み込みエラー</div>
          </div>
        ) : (
          <>
            {(isLoading || !audioReady) && (
              <div className="flex flex-col items-center justify-center h-32 mb-4 space-y-2">
                <div className="text-lg text-gray-600">音声ファイルを読み込み中...</div>
                <div className="text-sm text-gray-500">
                  {audioMetadata.name || `音声 ${currentAudioIndex + 1}/${AUDIO_FILES.length}`}
                </div>
                <div className="animate-pulse flex space-x-1">
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                </div>
              </div>
            )}

            {/* 波形エディタ */}
            <AudioWaveform
              ref={waveformRef}
              audioUrl={audioFile}
              markings={markings}
              onAddMarking={addMarking}
              onRemoveMarking={removeMarking}
              onUpdateMarking={updateMarking}
              currentTime={currentTime}
              onTimeUpdate={setCurrentTime}
              isPlaying={isPlaying}
              onPlayPause={setIsPlaying}
              zoomLevel={zoomLevel}
              onZoomChange={setZoomLevel}
              onMetadataLoad={(metadata) => {
                console.log('Metadata loaded:', metadata);
                setAudioMetadata(metadata);
                setIsLoading(false);
                setAudioReady(true);
              }}
            />

            {/* 再生コントロール */}
            <AudioControls
              isPlaying={isPlaying}
              onPlayPause={setIsPlaying}
              currentTime={currentTime}
              duration={audioMetadata.duration}
              onSeek={setCurrentTime}
              playbackRate={playbackRate}
              onPlaybackRateChange={setPlaybackRate}
              waveformRef={waveformRef}
            />

            {/* マーキングリスト */}
            <MarkingsList
              markings={markings}
              onRemove={removeMarking}
              onEdit={updateMarking}
              onSeek={(time) => {
                setCurrentTime(time);
                waveformRef.current?.seekTo(time);
              }}
              onPlaySection={(marking) => {
                // Play only the marked section
                waveformRef.current?.playSection(marking.start, marking.end);
                setPlayingSection(marking.id);
              }}
              playingSection={playingSection}
              duration={audioMetadata.duration}
            />

            {/* スコアスライダー */}
            <ScoreSlider
              score={score}
              onChange={setScore}
            />

            {/* アクションボタン */}
            <div className="mt-6 flex justify-end">
              <div className="space-x-3 flex items-center">
                <button
                  onClick={handlePrevious}
                  disabled={currentAudioIndex === 0}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  戻る
                </button>
                <button
                  onClick={handleSave}
                  className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
                >
                  保存のみ
                </button>
                <button
                  onClick={handleNext}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  完了して次へ (Enter)
                </button>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
