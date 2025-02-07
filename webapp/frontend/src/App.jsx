import { useState } from 'react'
import './App.css'

// API URL based on environment
const API_URL = import.meta.env.PROD 
  ? '/api' // Production: Use nginx proxy
  : 'http://localhost:8000' // Development: Direct connection

function App() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [documents, setDocuments] = useState(null)
  const handleSubmit = async () => {
    try {
      const res = await fetch(`${API_URL}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
        setResults(null)
      } else {
        console.log(data)
        // setDocuments(data.documents)
        setResults(data.results)
        setError(null)
      }
    } catch (error) {
      console.error('Error:', error)
      setError('Error submitting text')
      setResults(null)
    }
  }
  console.log(results, "results")
  return (
    <div className="container">
      <img 
        src="/paint.png" 
        alt="Paint Logo" 
        style={{ width: '100%' }}
      />
      <div className="input-container">

        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="How can I help you"
          className="text-input"
        />
        <button onClick={handleSubmit} className="submit-button">
          Search
        </button>
      </div>
      
      {error && <p className="error">{error}</p>}
      
      {results && results.length > 0 && (
        <div className="results-container">
          <ul className="similar-words">
            {results.map(({distance, document}, idx) => (
              <li style={{color: "black"}} key={idx + document}>
                {document} <span className="similarity">({distance} distance)</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
