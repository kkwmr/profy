import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

const WaveformEditor = forwardRef(({
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
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const regionsRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [tempRegion, setTempRegion] = useState(null);

  useImperativeHandle(ref, () => ({
    seekTo: (time) => {
      if (wavesurferRef.current) {
        const progress = time / wavesurferRef.current.getDuration();
        wavesurferRef.current.seekTo(progress);
      }
    },
    play: () => wavesurferRef.current?.play(),
    pause: () => wavesurferRef.current?.pause(),
    playPause: () => wavesurferRef.current?.playPause()
  }));

  // WaveSurfer初期化
  useEffect(() => {
    if (!waveformRef.current || !audioUrl) return;

    // 既存のインスタンスをクリーンアップ
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    // WaveSurfer作成（高速化設定）
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#6B7280',
      progressColor: '#3B82F6',
      cursorColor: '#EF4444',
      barWidth: 6,  // さらに太くして描画を高速化
      barGap: 2,    // バー間のギャップを追加
      barRadius: 2,
      cursorWidth: 2,
      height: 200,
      normalize: true,
      responsive: true,
      interact: true,
      minPxPerSec: 5,  // 最小限まで下げて高速化
      scrollParent: false,  // スクロールを無効化して高速化
      hideScrollbar: true,
      backend: 'MediaElement',  // MediaElementバックエンドで高速化
      mediaControls: false,
      plugins: []
    });

    // Regionsプラグイン初期化
    const regions = wavesurfer.registerPlugin(RegionsPlugin.create());
    
    wavesurferRef.current = wavesurfer;
    regionsRef.current = regions;

    // イベントリスナー
    // ローディング状態の早期解除
    wavesurfer.on('decode', () => {
      // デコード完了時点で使用可能にする
      setIsReady(true);
      const duration = wavesurfer.getDuration();
      onMetadataLoad(prev => ({ ...prev, duration }));
    });
    
    wavesurfer.on('ready', () => {
      // 完全に準備完了
      setIsReady(true);
      const duration = wavesurfer.getDuration();
      onMetadataLoad(prev => ({ ...prev, duration }));
    });

    wavesurfer.on('audioprocess', () => {
      onTimeUpdate(wavesurfer.getCurrentTime());
    });

    wavesurfer.on('play', () => onPlayPause(true));
    wavesurfer.on('pause', () => onPlayPause(false));

    // 音声ファイル読み込み
    wavesurfer.load(audioUrl);

    return () => {
      wavesurfer.destroy();
    };
  }, [audioUrl]);

  // マウスドラッグによる範囲選択
  useEffect(() => {
    if (!waveformRef.current || !regionsRef.current) return;

    const container = waveformRef.current;
    let startX = null;
    let startTime = null;
    let currentRegion = null;

    const getTimeFromX = (x) => {
      const rect = container.getBoundingClientRect();
      const relativeX = x - rect.left + container.scrollLeft;
      const width = container.scrollWidth;
      const duration = wavesurferRef.current.getDuration();
      return (relativeX / width) * duration;
    };

    const handleMouseDown = (e) => {
      // 左クリックのみ
      if (e.button !== 0) return;
      
      // 既存のリージョン上でなければ新規作成
      const clickTime = getTimeFromX(e.clientX);
      const existingRegion = Object.values(regionsRef.current.regions).find(
        r => clickTime >= r.start && clickTime <= r.end
      );
      
      if (!existingRegion) {
        e.preventDefault();
        startX = e.clientX;
        startTime = clickTime;
        setIsDragging(true);
        setDragStart(clickTime);
        
        // 仮のリージョンを作成
        currentRegion = regionsRef.current.addRegion({
          start: startTime,
          end: startTime,
          color: 'rgba(239, 68, 68, 0.3)',
          drag: false,
          resize: false
        });
      }
    };

    const handleMouseMove = (e) => {
      if (!startX || !currentRegion) return;
      
      const currentTime = getTimeFromX(e.clientX);
      const start = Math.min(startTime, currentTime);
      const end = Math.max(startTime, currentTime);
      
      // リージョンを更新
      currentRegion.setOptions({
        start,
        end
      });
      
      setTempRegion({ start, end });
    };

    const handleMouseUp = (e) => {
      if (!startX || !currentRegion) return;
      
      const endTime = getTimeFromX(e.clientX);
      const start = Math.min(startTime, endTime);
      const end = Math.max(startTime, endTime);
      
      // 最小長さチェック（0.5秒以上）
      if (end - start >= 0.5) {
        onAddMarking(start, end);
        currentRegion.setOptions({
          color: 'rgba(239, 68, 68, 0.6)',
          drag: true,
          resize: true
        });
      } else {
        currentRegion.remove();
      }
      
      startX = null;
      startTime = null;
      currentRegion = null;
      setIsDragging(false);
      setDragStart(null);
      setTempRegion(null);
    };

    container.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      container.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isReady, onAddMarking]);

  // マーキングの同期
  useEffect(() => {
    if (!regionsRef.current) return;
    
    // 既存のリージョンをクリア
    regionsRef.current.clearRegions();
    
    // マーキングからリージョンを作成
    markings.forEach(marking => {
      const region = regionsRef.current.addRegion({
        id: marking.id,
        start: marking.start,
        end: marking.end,
        color: 'rgba(239, 68, 68, 0.6)',
        drag: true,
        resize: true
      });
      
      // リージョンのイベントハンドラ
      region.on('update-end', () => {
        onUpdateMarking(marking.id, region.start, region.end);
      });
      
      region.on('dblclick', () => {
        onRemoveMarking(marking.id);
        region.remove();
      });
    });
  }, [markings, isReady]);

  // ズーム処理
  useEffect(() => {
    if (wavesurferRef.current && isReady) {
      wavesurferRef.current.zoom(zoomLevel);
    }
  }, [zoomLevel, isReady]);

  // 再生制御
  useEffect(() => {
    if (!wavesurferRef.current || !isReady) return;
    
    if (isPlaying) {
      wavesurferRef.current.play();
    } else {
      wavesurferRef.current.pause();
    }
  }, [isPlaying, isReady]);

  // キーボードショートカット
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (!wavesurferRef.current) return;
      
      switch(e.code) {
        case 'Space':
          e.preventDefault();
          wavesurferRef.current.playPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          const currentTime = wavesurferRef.current.getCurrentTime();
          wavesurferRef.current.seekTo((currentTime - 5) / wavesurferRef.current.getDuration());
          break;
        case 'ArrowRight':
          e.preventDefault();
          const current = wavesurferRef.current.getCurrentTime();
          wavesurferRef.current.seekTo((current + 5) / wavesurferRef.current.getDuration());
          break;
        default:
          break;
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isReady]);

  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      {/* ズームコントロール */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-700">ズーム:</span>
          <button
            onClick={() => onZoomChange(Math.max(10, zoomLevel - 10))}
            className="p-1 hover:bg-gray-100 rounded"
            title="ズームアウト"
          >
            <ZoomOut size={18} />
          </button>
          <input
            type="range"
            min="10"
            max="200"
            value={zoomLevel}
            onChange={(e) => onZoomChange(parseInt(e.target.value))}
            className="w-32"
          />
          <button
            onClick={() => onZoomChange(Math.min(200, zoomLevel + 10))}
            className="p-1 hover:bg-gray-100 rounded"
            title="ズームイン"
          >
            <ZoomIn size={18} />
          </button>
          <button
            onClick={() => onZoomChange(50)}
            className="p-1 hover:bg-gray-100 rounded"
            title="フィット"
          >
            <Maximize2 size={18} />
          </button>
        </div>
        
        {isDragging && (
          <div className="text-sm text-gray-600">
            選択中: {dragStart?.toFixed(1)}s - 
            {tempRegion ? tempRegion.end.toFixed(1) : '...'}s
          </div>
        )}
        
        {!isReady && (
          <div className="text-sm text-gray-500">
            読み込み中...
          </div>
        )}
      </div>

      {/* 波形表示エリア */}
      <div 
        ref={waveformRef}
        className="waveform-container"
        style={{
          position: 'relative',
          overflow: 'auto',
          border: '1px solid #e5e7eb',
          borderRadius: '0.375rem',
          cursor: isDragging ? 'crosshair' : 'pointer'
        }}
      />
      
      {/* タイムライン */}
      <div className="mt-2 px-2 flex justify-between text-xs text-gray-500">
        <span>0:00</span>
        <span>{currentTime ? formatTime(currentTime) : '0:00'}</span>
        <span>{wavesurferRef.current ? formatTime(wavesurferRef.current.getDuration()) : '0:00'}</span>
      </div>
      
      {/* 操作ヒント */}
      <div className="mt-3 text-xs text-gray-500 flex justify-center space-x-4">
        <span>クリック: 再生位置指定</span>
        <span>ドラッグ: 範囲選択</span>
        <span>ダブルクリック: 削除</span>
        <span>Space: 再生/停止</span>
      </div>
    </div>
  );
});

// 時間フォーマット関数
const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

WaveformEditor.displayName = 'WaveformEditor';

export default WaveformEditor;