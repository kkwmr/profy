import React from 'react';

const ScoreSlider = ({ score, onChange }) => {
  const getScoreColor = (value) => {
    if (value >= 8) return 'text-green-600';
    if (value >= 6) return 'text-yellow-600';
    if (value >= 4) return 'text-orange-600';
    return 'text-red-600';
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mt-4">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">総合評価</h3>
          <p className="text-xs text-gray-500 mt-1">1 (Bad) - 10 (Good)</p>
        </div>
        <span className={`text-2xl font-bold ${getScoreColor(score)}`}>
          {score.toFixed(1)} / 10
        </span>
      </div>
      
      <div className="relative">
        <input
          type="range"
          min="0"
          max="10"
          step="0.5"
          value={score}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${score * 10}%, #E5E7EB ${score * 10}%, #E5E7EB 100%)`
          }}
        />
        
        {/* 目盛り */}
        <div className="flex justify-between mt-2">
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(num => (
            <span key={num} className="text-xs text-gray-500">
              {num}
            </span>
          ))}
        </div>
      </div>
      
      {/* クイック選択ボタン */}
      <div className="mt-4 flex justify-center space-x-2">
        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(num => (
          <button
            key={num}
            onClick={() => onChange(num)}
            className={`w-8 h-8 rounded ${
              Math.floor(score) === num
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            } text-sm font-medium transition-colors`}
          >
            {num}
          </button>
        ))}
      </div>
    </div>
  );
};

export default ScoreSlider;