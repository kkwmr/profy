import React from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

const AudioControls = ({
  isPlaying,
  onPlayPause,
  currentTime,
  duration,
  onSeek,
  playbackRate,
  onPlaybackRateChange,
  waveformRef
}) => {
  
  const handleSkipBack = () => {
    const newTime = Math.max(0, currentTime - 5);
    onSeek(newTime);
    waveformRef.current?.seekTo(newTime);
  };
  
  const handleSkipForward = () => {
    const newTime = Math.min(duration, currentTime + 5);
    onSeek(newTime);
    waveformRef.current?.seekTo(newTime);
  };
  
  const handlePlayPause = () => {
    waveformRef.current?.playPause();
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mt-4">
      <div className="flex items-center justify-center space-x-4">
        {/* スキップバック */}
        <button
          onClick={handleSkipBack}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          title="5秒戻る"
        >
          <SkipBack size={20} />
        </button>
        
        {/* 再生/一時停止 */}
        <button
          onClick={handlePlayPause}
          className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full transition-colors"
          title={isPlaying ? '一時停止' : '再生'}
        >
          {isPlaying ? <Pause size={24} /> : <Play size={24} />}
        </button>
        
        {/* スキップフォワード */}
        <button
          onClick={handleSkipForward}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          title="5秒進む"
        >
          <SkipForward size={20} />
        </button>
        
        {/* 再生速度 */}
        <div className="flex items-center space-x-2 ml-6">
          <span className="text-sm text-gray-600">速度:</span>
          <select
            value={playbackRate}
            onChange={(e) => {
              const rate = parseFloat(e.target.value);
              onPlaybackRateChange(rate);
              if (waveformRef.current?.wavesurfer) {
                waveformRef.current.wavesurfer.setPlaybackRate(rate);
              }
            }}
            className="px-2 py-1 border border-gray-300 rounded text-sm"
          >
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1">1.0x</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2.0x</option>
          </select>
        </div>
      </div>
      
      {/* プログレスバー */}
      <div className="mt-4 px-4">
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600 w-12">
            {formatTime(currentTime)}
          </span>
          <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-600 transition-all duration-100"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
          </div>
          <span className="text-sm text-gray-600 w-12 text-right">
            {formatTime(duration)}
          </span>
        </div>
      </div>
    </div>
  );
};

const formatTime = (seconds) => {
  if (!seconds || isNaN(seconds)) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default AudioControls;