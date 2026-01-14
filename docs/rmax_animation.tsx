import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, SkipForward } from 'lucide-react';

const RmaxAnimation = () => {
  const gridSize = 10;
  const numActions = 4; // up, down, left, right
  const optimisticValue = 10;
  const gamma = 0.9;
  const viIterations = 20; // Value iteration iterations
  
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [agentPos, setAgentPos] = useState(gridSize * (gridSize - 1)); // Bottom left
  const [knownStateActions, setKnownStateActions] = useState(new Set());
  const [Q, setQ] = useState(() => {
    const initQ = {};
    for (let i = 0; i < gridSize * gridSize; i++) {
      initQ[i] = Array(numActions).fill(optimisticValue);
    }
    return initQ;
  });
  
  // Convert state index to (row, col)
  const idxToPos = (idx) => ({
    row: Math.floor(idx / gridSize),
    col: idx % gridSize
  });
  
  // Convert (row, col) to state index
  const posToIdx = (row, col) => row * gridSize + col;
  
  // Get next state given current state and action
  const getNextState = (state, action) => {
    const { row, col } = idxToPos(state);
    let newRow = row, newCol = col;
    
    switch(action) {
      case 0: newRow = Math.max(0, row - 1); break; // up
      case 1: newRow = Math.min(gridSize - 1, row + 1); break; // down
      case 2: newCol = Math.max(0, col - 1); break; // left
      case 3: newCol = Math.min(gridSize - 1, col + 1); break; // right
    }
    
    return posToIdx(newRow, newCol);
  };
  
  // Run value iteration
  const runValueIteration = (known) => {
    let newQ = {};
    for (let s = 0; s < gridSize * gridSize; s++) {
      newQ[s] = Array(numActions).fill(0);
    }
    
    // Initialize Q-values
    for (let s = 0; s < gridSize * gridSize; s++) {
      for (let a = 0; a < numActions; a++) {
        const key = `${s},${a}`;
        if (known.has(key)) {
          // Known state-action: use value iteration
          newQ[s][a] = 0; // Will be updated in VI loop
        } else {
          // Unknown state-action: optimistic value
          newQ[s][a] = optimisticValue;
        }
      }
    }
    
    // Value iteration
    for (let iter = 0; iter < viIterations; iter++) {
      const oldQ = JSON.parse(JSON.stringify(newQ));
      
      for (let s = 0; s < gridSize * gridSize; s++) {
        for (let a = 0; a < numActions; a++) {
          const key = `${s},${a}`;
          if (known.has(key)) {
            const nextState = getNextState(s, a);
            const reward = 0; // No rewards in this grid
            const maxNextQ = Math.max(...oldQ[nextState]);
            newQ[s][a] = reward + gamma * maxNextQ;
          }
        }
      }
    }
    
    return newQ;
  };
  
  // Get greedy action (break ties randomly)
  const getGreedyAction = (state, qValues) => {
    let maxQ = -Infinity;
    let bestActions = [];
    
    for (let a = 0; a < numActions; a++) {
      if (qValues[state][a] > maxQ) {
        maxQ = qValues[state][a];
        bestActions = [a];
      } else if (qValues[state][a] === maxQ) {
        bestActions.push(a);
      }
    }
    
    // Randomly select from tied actions
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  };
  
  const step = () => {
    const currentState = agentPos;
    const greedyAction = getGreedyAction(currentState, Q);
    const nextState = getNextState(currentState, greedyAction);
    
    // Mark state-action as known
    const key = `${currentState},${greedyAction}`;
    const newKnown = new Set(knownStateActions);
    newKnown.add(key);
    
    // Run value iteration with updated known set
    const newQ = runValueIteration(newKnown);
    
    setKnownStateActions(newKnown);
    setQ(newQ);
    setAgentPos(nextState);
    setFrame(frame + 1);
  };
  
  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(() => {
        step();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, frame, agentPos, Q, knownStateActions]);
  
  const reset = () => {
    setFrame(0);
    setIsPlaying(false);
    setAgentPos(gridSize * (gridSize - 1));
    setKnownStateActions(new Set());
    const initQ = {};
    for (let i = 0; i < gridSize * gridSize; i++) {
      initQ[i] = Array(numActions).fill(optimisticValue);
    }
    setQ(initQ);
  };
  
  // Get max Q-value for each state
  const getMaxQ = (stateIdx) => {
    return Math.max(...Q[stateIdx]);
  };
  
  // Color mapping
  const getColor = (value) => {
    const normalized = value / optimisticValue;
    const r = Math.floor(255 * (1 - normalized));
    const g = Math.floor(200 * normalized);
    const b = 50;
    return `rgb(${r}, ${g}, ${b})`;
  };
  
  // Count known state-actions per state
  const getKnownActionsCount = (state) => {
    let count = 0;
    for (let a = 0; a < numActions; a++) {
      if (knownStateActions.has(`${state},${a}`)) count++;
    }
    return count;
  };
  
  return (
    <div className="flex flex-col items-center gap-6 p-8 bg-gray-50 min-h-screen">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">R-max Algorithm</h1>
        <p className="text-gray-600">Steps: {frame} | Known state-actions: {knownStateActions.size}</p>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
          {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
            const maxQ = getMaxQ(idx);
            const isAgent = agentPos === idx;
            const knownActions = getKnownActionsCount(idx);
            
            return (
              <div
                key={idx}
                className="w-12 h-12 flex items-center justify-center text-xs font-semibold border transition-all duration-300 relative"
                style={{
                  backgroundColor: getColor(maxQ),
                  borderColor: isAgent ? '#3b82f6' : '#d1d5db',
                  borderWidth: isAgent ? '3px' : '1px'
                }}
                title={`State ${idx}: max Q = ${maxQ.toFixed(2)}, known actions: ${knownActions}/4`}
              >
                <span className={isAgent ? 'text-blue-600 font-bold' : ''}>
                  {maxQ.toFixed(1)}
                </span>
                {knownActions > 0 && (
                  <div className="absolute top-0 right-0 w-2 h-2 bg-green-500 rounded-full"></div>
                )}
              </div>
            );
          })}
        </div>
      </div>
      
      <div className="flex gap-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        
        <button
          onClick={step}
          className="flex items-center gap-2 px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
        >
          <SkipForward size={20} />
          Step
        </button>
        
        <button
          onClick={reset}
          className="flex items-center gap-2 px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          <RotateCcw size={20} />
          Reset
        </button>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-2xl">
        <h3 className="font-bold text-lg mb-3">R-max Algorithm</h3>
        <div className="space-y-2 text-sm">
          <p>• Agent starts at bottom left corner</p>
          <p>• Takes greedy action based on Q-values</p>
          <p>• Unknown state-actions have optimistic Q = {optimisticValue}</p>
          <p>• When a state-action becomes known, value iteration updates all Q-values</p>
          <p>• Green dot = at least one action known for that state</p>
          <p>• Blue border = current agent position</p>
        </div>
      </div>
    </div>
  );
};

export default RmaxAnimation;