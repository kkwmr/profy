import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';

const AudioWaveform = forwardRef(({
  audioUrl,
  markings,
  onAddMarking,
  onRemoveMarking,
  onUpdateMarking,
  currentTime,
  onTimeUpdate,
  isPlaying,
  onPlayPause,
  zoomLevel,
  onZoomChange,
  onMetadataLoad
}, ref) => {
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [peaks, setPeaks] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [resizingMarking, setResizingMarking] = useState(null);
  const [resizeType, setResizeType] = useState(null); // 'start' or 'end'
  const [sectionEndTime, setSectionEndTime] = useState(null); // For section playback

  useImperativeHandle(ref, () => ({
    seekTo: (time) => {
      if (audioRef.current) {
        audioRef.current.currentTime = time;
      }
    },
    play: () => audioRef.current?.play(),
    pause: () => audioRef.current?.pause(),
    playPause: () => {
      if (audioRef.current) {
        if (audioRef.current.paused) {
          audioRef.current.play();
        } else {
          audioRef.current.pause();
        }
      }
    },
    playSection: (startTime, endTime) => {
      if (audioRef.current) {
        audioRef.current.currentTime = startTime;
        setSectionEndTime(endTime);
        audioRef.current.play();
        if (onPlayPause) onPlayPause(true);
      }
    }
  }));

  // 初期化と音声読み込み
  useEffect(() => {
    console.log('AudioWaveform: Initializing with URL:', audioUrl);
    
    if (!audioUrl || !audioRef.current) {
      console.log('AudioWaveform: No URL or audio element');
      return;
    }

    const audio = audioRef.current;
    
    // イベントハンドラー
    const handleLoadedMetadata = () => {
      console.log('AudioWaveform: Metadata loaded, duration:', audio.duration);
      const dur = audio.duration || 0;
      setDuration(dur);
      
      // 親コンポーネントに通知
      if (onMetadataLoad) {
        onMetadataLoad({
          name: audioUrl.split('/').pop(),
          duration: dur,
          sampleRate: 48000
        });
      }
      
      // 波形データ生成（実データからピーク抽出）
      generateWaveformData();
    };

    const handleTimeUpdate = () => {
      if (onTimeUpdate) {
        onTimeUpdate(audio.currentTime);
      }
      
      // Stop at section end if playing a section
      if (sectionEndTime && audio.currentTime >= sectionEndTime) {
        audio.pause();
        setSectionEndTime(null);
        if (onPlayPause) onPlayPause(false);
      }
    };

    const handlePlay = () => {
      if (onPlayPause) onPlayPause(true);
    };

    const handlePause = () => {
      setSectionEndTime(null); // Clear section end when paused
      if (onPlayPause) onPlayPause(false);
    };

    // 波形データを生成（Web Audio API でデコード）
    const generateWaveformData = async () => {
      try {
        const res = await fetch(audioUrl, { cache: 'force-cache' });
        const arrayBuffer = await res.arrayBuffer();
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioCtx({ sampleRate: 48000 });
        const decoded = await ctx.decodeAudioData(arrayBuffer);
        setAudioBuffer(decoded);

        // 初期ピーク解像度を上げる（より詳細な波形）
        const targetBars = 800;  // 300から800に増やす
        const newPeaks = computePeaks(decoded, targetBars);
        setPeaks(newPeaks);

        // すぐ閉じる（iOS 自動再生制限の回避用）
        ctx.close?.();
      } catch (e) {
        console.error('Failed to decode audio for waveform:', e);
        // フォールバック: ランダム波形（詳細度を合わせる）
        const fallback = [];
        for (let i = 0; i < 800; i++) fallback.push(0.2 + Math.random() * 0.6);
        setPeaks(fallback);
      }
    };

    const computePeaks = (buffer, length) => {
      const { numberOfChannels, duration } = buffer;
      const sampleRate = buffer.sampleRate;
      const totalSamples = Math.floor(duration * sampleRate);
      const samplesPerBucket = Math.max(1, Math.floor(totalSamples / length));

      const peaks = new Array(length).fill(0);
      for (let ch = 0; ch < numberOfChannels; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
          const start = i * samplesPerBucket;
          const end = Math.min(start + samplesPerBucket, totalSamples);
          let peak = 0;
          for (let s = start; s < end; s++) {
            const v = Math.abs(data[s] || 0);
            if (v > peak) peak = v;
          }
          if (peak > peaks[i]) peaks[i] = peak; // チャンネル間で最大
        }
      }
      // 正規化
      const max = peaks.reduce((m, v) => (v > m ? v : m), 0.0001);
      return peaks.map(v => v / max);
    };

    // イベントリスナー登録
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);

    // 音声をロード
    console.log('AudioWaveform: Setting source and loading');
    audio.src = audioUrl;
    audio.load();

    // 強制的に初期化（フォールバック）
    setTimeout(() => {
      console.log('AudioWaveform: Fallback initialization');
      if (duration === 0) {
        handleLoadedMetadata();
      }
    }, 100);

    // クリーンアップ
    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
    };
  }, [audioUrl]); // audioUrlのみに依存

  // セクション再生の管理
  useEffect(() => {
    if (!audioRef.current || !sectionEndTime) return;
    
    const handleSectionTimeUpdate = () => {
      if (audioRef.current && audioRef.current.currentTime >= sectionEndTime) {
        audioRef.current.pause();
        setSectionEndTime(null);
        if (onPlayPause) onPlayPause(false);
      }
    };
    
    audioRef.current.addEventListener('timeupdate', handleSectionTimeUpdate);
    
    return () => {
      if (audioRef.current) {
        audioRef.current.removeEventListener('timeupdate', handleSectionTimeUpdate);
      }
    };
  }, [sectionEndTime, onPlayPause]);

  // 再生/一時停止の同期
  useEffect(() => {
    if (!audioRef.current) return;
    
    if (isPlaying && audioRef.current.paused) {
      audioRef.current.play().catch(e => console.log('Play failed:', e));
    } else if (!isPlaying && !audioRef.current.paused) {
      audioRef.current.pause();
    }
  }, [isPlaying]);

  // 波形描画
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // 高解像度レンダリング
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 800;
    const cssHeight = 200;
    const width = Math.floor(cssWidth * dpr);
    const height = Math.floor(cssHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    // クリア
    ctx.clearRect(0, 0, width, height);
    
    // 背景
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, width, height);
    
    // 波形描画
    if (peaks.length > 0) {
      const centerY = height / 2;
      const progress = audioRef.current ? (audioRef.current.currentTime / (duration || 1)) : 0;
      const playedIndex = Math.floor(peaks.length * progress);
      const pxPerBar = width / peaks.length;
      
      // バーの幅と間隔を動的に調整
      let barWidth, barGap;
      if (pxPerBar >= 4) {
        barWidth = Math.floor(pxPerBar * 0.8);
        barGap = Math.floor(pxPerBar * 0.2);
      } else if (pxPerBar >= 2) {
        barWidth = Math.floor(pxPerBar * 0.9);
        barGap = Math.max(0.5, pxPerBar * 0.1);
      } else {
        barWidth = Math.max(1, pxPerBar);
        barGap = 0;
      }

      for (let i = 0; i < peaks.length; i++) {
        const x = i * pxPerBar;
        const barHeight = Math.max(2, peaks[i] * height * 0.85);
        ctx.fillStyle = i <= playedIndex ? '#3b82f6' : '#6b7280';
        
        // より滑らかな描画
        if (barWidth >= 1) {
          ctx.fillRect(x, centerY - barHeight / 2, barWidth, barHeight);
        } else {
          // 1px未満の場合は線で描画
          ctx.beginPath();
          ctx.moveTo(x, centerY - barHeight / 2);
          ctx.lineTo(x, centerY + barHeight / 2);
          ctx.strokeStyle = i <= playedIndex ? '#3b82f6' : '#6b7280';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    
    // マーキング描画
    if (markings && duration > 0) {
      markings.forEach(marking => {
        const startX = (marking.start / duration) * width;
        const endX = (marking.end / duration) * width;
        
        // マーキングエリア
        ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
        ctx.fillRect(startX, 0, endX - startX, height);
        
        // 境界線
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startX, 0);
        ctx.lineTo(startX, height);
        ctx.moveTo(endX, 0);
        ctx.lineTo(endX, height);
        ctx.stroke();
        
        // リサイズハンドル（つまみ）
        const handleSize = 8;
        const handleY = height / 2;
        
        // 左ハンドル
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(startX - handleSize/2, handleY - handleSize/2, handleSize, handleSize);
        
        // 右ハンドル
        ctx.fillRect(endX - handleSize/2, handleY - handleSize/2, handleSize, handleSize);
      });
    }
    
    // ドラッグ中の選択範囲
    if (isDragging && dragStart !== null) {
      const startX = dragStart * width;
      const endX = (audioRef.current ? (audioRef.current.currentTime / duration) : 0) * width;
      
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.fillRect(Math.min(startX, endX), 0, Math.abs(endX - startX), height);
    }
    
    // 再生位置
    if (audioRef.current && duration > 0) {
      const progress = audioRef.current.currentTime / duration;
      const x = progress * width;
      
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }, [peaks, markings, isDragging, dragStart, currentTime, duration, zoomLevel]);

  // ズームは表示スケールのみ（ピークは固定解像度）

  // マウスイベント
  const handleMouseDown = (e) => {
    if (!canvasRef.current || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickTime = (x / rect.width) * duration;
    
    // ダブルクリックで削除
    if (e.detail === 2) {
      const marking = markings.find(m => 
        clickTime >= m.start && clickTime <= m.end
      );
      if (marking && onRemoveMarking) {
        onRemoveMarking(marking.id);
      }
      return;
    }
    
    // ハンドルのチェック（リサイズ用）
    const handleThreshold = 5 / rect.width * duration; // 5px分の時間
    for (const marking of markings) {
      if (Math.abs(clickTime - marking.start) < handleThreshold) {
        setResizingMarking(marking);
        setResizeType('start');
        return;
      }
      if (Math.abs(clickTime - marking.end) < handleThreshold) {
        setResizingMarking(marking);
        setResizeType('end');
        return;
      }
    }
    
    // 新規マーキングの開始
    setIsDragging(true);
    setDragStart(clickTime / duration);
  };

  const handleMouseMove = (e) => {
    if (!canvasRef.current || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const currentTime = (x / rect.width) * duration;
    
    // リサイズ中の処理
    if (resizingMarking && resizeType && onUpdateMarking) {
      e.preventDefault();
      if (resizeType === 'start') {
        const newStart = Math.min(currentTime, resizingMarking.end - 0.1);
        onUpdateMarking(resizingMarking.id, Math.max(0, newStart), resizingMarking.end);
      } else if (resizeType === 'end') {
        const newEnd = Math.max(currentTime, resizingMarking.start + 0.1);
        onUpdateMarking(resizingMarking.id, resizingMarking.start, Math.min(duration, newEnd));
      }
    }
    
    // カーソルスタイルの変更
    const handleThreshold = 5 / rect.width * duration;
    let cursorSet = false;
    for (const marking of markings) {
      if (Math.abs(currentTime - marking.start) < handleThreshold || 
          Math.abs(currentTime - marking.end) < handleThreshold) {
        canvasRef.current.style.cursor = 'ew-resize';
        cursorSet = true;
        break;
      }
    }
    if (!cursorSet) {
      canvasRef.current.style.cursor = 'crosshair';
    }
  };

  const handleMouseUp = (e) => {
    // リサイズ終了
    if (resizingMarking) {
      setResizingMarking(null);
      setResizeType(null);
      return;
    }
    
    // 新規マーキング作成
    if (!isDragging || !canvasRef.current || !duration || dragStart === null) {
      setIsDragging(false);
      setDragStart(null);
      return;
    }
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const endTime = (x / rect.width) * duration;
    const startTime = dragStart * duration;
    
    if (Math.abs(endTime - startTime) > 0.1 && onAddMarking) {
      onAddMarking(Math.min(startTime, endTime), Math.max(startTime, endTime));
    }
    
    setIsDragging(false);
    setDragStart(null);
  };

  const handleClick = (e) => {
    if (isDragging) return;
    if (!canvasRef.current || !audioRef.current || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickTime = (x / rect.width) * duration;
    
    audioRef.current.currentTime = clickTime;
    if (onTimeUpdate) onTimeUpdate(clickTime);
  };

  return (
    <div className="space-y-4">
      {/* 非表示の音声要素 */}
      <audio ref={audioRef} className="hidden" />
      
      {/* 波形表示 */}
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-700">波形エディタ</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onZoomChange && onZoomChange(Math.max(100, zoomLevel - 25))}
              className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-50 transition-colors"
              disabled={zoomLevel <= 100}
              title="ズームアウト"
            >
              <ZoomOut size={18} />
            </button>
            <span className="text-sm text-gray-600 font-medium min-w-[3rem] text-center inline-block">
              {zoomLevel}%
            </span>
            <button
              onClick={() => onZoomChange && onZoomChange(Math.min(500, zoomLevel + 25))}
              className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-50 transition-colors"
              disabled={zoomLevel >= 500}
              title="ズームイン"
            >
              <ZoomIn size={18} />
            </button>
          </div>
        </div>
        
        <div className="relative bg-gray-50 rounded overflow-x-auto overflow-y-hidden" 
             style={{ 
               maxHeight: '220px',
               scrollbarWidth: 'thin',
               scrollbarColor: '#9ca3af #f3f4f6'
             }}>
          <canvas
            ref={canvasRef}
            width={800}
            height={200}
            className="cursor-crosshair block"
            style={{ 
              width: `${Math.max(100, zoomLevel)}%`,
              height: '200px'
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
          />
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          <div>ドラッグで問題箇所を選択（複数選択可） | 端をドラッグでサイズ調整 | ダブルクリックで削除 | クリックでシーク</div>
        </div>
      </div>
    </div>
  );
});

AudioWaveform.displayName = 'AudioWaveform';

export default AudioWaveform;
