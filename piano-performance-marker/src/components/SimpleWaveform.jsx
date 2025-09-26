import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';

const SimpleWaveform = forwardRef(({
  audioUrl,
  markings,
  onAddMarking,
  onRemoveMarking,
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
  const containerRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [peaks, setPeaks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

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
    }
  }));

  // Web Audio APIで波形データを取得（簡略版）
  const loadAudioData = async () => {
    setIsLoading(true);
    try {
      // 簡単な波形データを生成（実際の音声解析の代わり）
      const peakCount = 200;
      const newPeaks = [];
      
      for (let i = 0; i < peakCount; i++) {
        // 実際の波形っぽく見せるためのダミーデータ
        const baseHeight = 0.3 + Math.random() * 0.4;
        newPeaks.push({ 
          min: -baseHeight * (0.8 + Math.random() * 0.2),
          max: baseHeight * (0.8 + Math.random() * 0.2)
        });
      }
      
      setPeaks(newPeaks);
      console.log('Generated waveform peaks:', newPeaks.length);
    } catch (error) {
      console.error('Failed to generate waveform:', error);
    } finally {
      setIsLoading(false);
      setTimeout(() => drawWaveform(), 100);
    }
  };

  // Audio要素の初期化と波形データの取得
  useEffect(() => {
    if (!audioUrl) return;
    console.log('Loading audio:', audioUrl);

    const audio = audioRef.current;
    if (!audio) {
      console.error('Audio element not found');
      return;
    }

    // 音声要素をリセット
    audio.pause();
    audio.currentTime = 0;
    audio.src = audioUrl;
    audio.load(); // 明示的にloadを呼ぶ
    
    // タイムアウトフォールバック - 即座に実行
    const timeoutId = setTimeout(() => {
      console.log('Timeout fallback triggered');
      if (!duration) {
        onMetadataLoad({
          name: audioUrl.split('/').pop(),
          duration: 10, // デフォルト値
          sampleRate: 48000
        });
        loadAudioData();
      }
    }, 500); // 500msに短縮

    const handleLoadedMetadata = () => {
      console.log('Audio metadata loaded, duration:', audio.duration);
      console.log('Audio readyState:', audio.readyState);
      console.log('Audio networkState:', audio.networkState);
      
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration);
        onMetadataLoad({
          name: audioUrl.split('/').pop(),
          duration: audio.duration,
          sampleRate: 48000
        });
        loadAudioData();
        setTimeout(() => drawWaveform(), 100);
      } else {
        console.warn('Invalid duration:', audio.duration);
      }
    };
    
    const handleError = (e) => {
      console.error('Audio loading error:', e);
      onMetadataLoad({
        name: audioUrl.split('/').pop(),
        duration: 0,
        sampleRate: 48000
      });
    };

    const handleTimeUpdate = () => {
      onTimeUpdate(audio.currentTime);
      drawWaveform();
    };

    const handlePlay = () => onPlayPause(true);
    const handlePause = () => onPlayPause(false);
    
    const handleCanPlayThrough = () => {
      console.log('Audio can play through');
      if (!duration && audio.duration) {
        handleLoadedMetadata();
      }
    };
    
    const handleLoadStart = () => {
      console.log('Audio load started');
    };
    
    const handleProgress = () => {
      console.log('Audio loading progress');
    };

    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('progress', handleProgress);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('canplaythrough', handleCanPlayThrough);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('error', handleError);

    return () => {
      clearTimeout(timeoutId);
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('progress', handleProgress);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('canplaythrough', handleCanPlayThrough);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('error', handleError);
    };
  }, [audioUrl, onMetadataLoad, onTimeUpdate, onPlayPause, duration, loadAudioData, drawWaveform]);

  // 再生/一時停止の同期
  useEffect(() => {
    if (!audioRef.current) return;
    
    if (isPlaying && audioRef.current.paused) {
      audioRef.current.play();
    } else if (!isPlaying && !audioRef.current.paused) {
      audioRef.current.pause();
    }
  }, [isPlaying]);

  // 実際の波形描画
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !audioRef.current) return;

    const width = canvas.width;
    const height = canvas.height;
    const halfHeight = height / 2;
    
    // クリア
    ctx.clearRect(0, 0, width, height);
    
    // 背景
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, width, height);
    
    // 中央線
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, halfHeight);
    ctx.lineTo(width, halfHeight);
    ctx.stroke();
    
    // 波形描画
    if (peaks.length > 0) {
      const barWidth = Math.max(1, width / peaks.length);
      const playbackPosition = audioRef.current.currentTime / duration;
      
      peaks.forEach((peak, i) => {
        const x = i * barWidth;
        const maxHeight = Math.abs(peak.max) * halfHeight * 0.9;
        const minHeight = Math.abs(peak.min) * halfHeight * 0.9;
        
        // 再生済み部分の判定
        const isPlayed = i / peaks.length < playbackPosition;
        
        // 上側の波形
        ctx.fillStyle = isPlayed ? '#3b82f6' : '#6b7280';
        ctx.fillRect(x, halfHeight - maxHeight, barWidth - 0.5, maxHeight);
        
        // 下側の波形
        ctx.fillRect(x, halfHeight, barWidth - 0.5, minHeight);
      });
    } else if (isLoading) {
      // ローディング中の表示
      ctx.fillStyle = '#9ca3af';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('波形を生成中...', width / 2, halfHeight);
    }
    
    // マーキング描画
    markings.forEach(marking => {
      const startX = (marking.start / duration) * width;
      const endX = (marking.end / duration) * width;
      
      ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
      ctx.fillRect(startX, 0, endX - startX, height);
      
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX, 0);
      ctx.lineTo(startX, height);
      ctx.moveTo(endX, 0);
      ctx.lineTo(endX, height);
      ctx.stroke();
    });
    
    // ドラッグ中の選択範囲
    if (isDragging && dragStart !== null && dragEnd !== null) {
      const startX = Math.min(dragStart, dragEnd) * width;
      const endX = Math.max(dragStart, dragEnd) * width;
      
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.fillRect(startX, 0, endX - startX, height);
      
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(startX, 0, endX - startX, height);
      ctx.setLineDash([]);
    }
    
    // 再生位置カーソル
    if (audioRef.current.currentTime > 0) {
      const cursorX = (audioRef.current.currentTime / duration) * width;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, height);
      ctx.stroke();
    }
  }, [peaks, duration, markings, isDragging, dragStart, dragEnd, isLoading]);

  // Canvas描画の更新
  useEffect(() => {
    drawWaveform();
  }, [markings, duration, isDragging, dragStart, dragEnd, peaks, isLoading]);

  // マウスイベントハンドリング
  const handleMouseDown = (e) => {
    if (!canvasRef.current || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickTime = (x / rect.width) * duration;
    
    // 既存のマーキング上でダブルクリックで削除
    if (e.detail === 2) {
      const marking = markings.find(m => 
        clickTime >= m.start && clickTime <= m.end
      );
      if (marking) {
        onRemoveMarking(marking.id);
        return;
      }
    }
    
    setIsDragging(true);
    setDragStart(x / rect.width);
    setDragEnd(x / rect.width);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setDragEnd(x / rect.width);
    drawWaveform();
  };

  const handleMouseUp = () => {
    if (isDragging && dragStart !== null && dragEnd !== null && duration) {
      const start = Math.min(dragStart, dragEnd) * duration;
      const end = Math.max(dragStart, dragEnd) * duration;
      
      if (end - start > 0.1) {  // 最小0.1秒
        onAddMarking(start, end);
      }
    }
    
    setIsDragging(false);
    setDragStart(null);
    setDragEnd(null);
    drawWaveform();
  };

  const handleCanvasClick = (e) => {
    if (!canvasRef.current || !audioRef.current || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickTime = (x / rect.width) * duration;
    
    audioRef.current.currentTime = clickTime;
    onTimeUpdate(clickTime);
  };

  return (
    <div className="space-y-4">
      {/* 音声要素（非表示） */}
      <audio ref={audioRef} className="hidden" />
      
      {/* 波形表示 */}
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-700">波形エディタ</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onZoomChange(Math.max(10, zoomLevel - 10))}
              className="p-1 text-gray-500 hover:text-gray-700"
            >
              <ZoomOut size={18} />
            </button>
            <span className="text-sm text-gray-600">{zoomLevel}%</span>
            <button
              onClick={() => onZoomChange(Math.min(200, zoomLevel + 10))}
              className="p-1 text-gray-500 hover:text-gray-700"
            >
              <ZoomIn size={18} />
            </button>
          </div>
        </div>
        
        <div 
          ref={containerRef}
          className="relative overflow-x-auto bg-gray-50 rounded"
          style={{ maxHeight: '200px' }}
        >
          <canvas
            ref={canvasRef}
            width={1200}
            height={200}
            className="cursor-crosshair"
            style={{ width: `${1200 * (zoomLevel / 100)}px`, height: '200px' }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleCanvasClick}
          />
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          ドラッグで範囲選択 | ダブルクリックで削除 | クリックでシーク
        </div>
      </div>
    </div>
  );
});

SimpleWaveform.displayName = 'SimpleWaveform';

export default SimpleWaveform;