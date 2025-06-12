import { useState } from 'react'
import './App.css'

// API URL based on environment
const API_URL = 'http://65.109.135.235:8090' // Production: Use nginx proxy
  // : 'http://localhost:8000' // Development: Direct connection

function App() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
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
  return (
    <div className="container">
      <div className="header-section">
        <h1 className="main-title">Semantic Search</h1>
        <p className="description">
          This is a two-tower neural network architecture with Google&apos;s word2vec embeddings used for the words. 
          The embeddings are passed through an RNN and the embeddings used are the final hidden layer of this RNN. 
          The dataset used was MSMarco&apos;s v1 (100k records) and the embeddings are saved with Faiss.
        </p>
        <div className="links-container" style={{display: 'flex', gap: '20px', justifyContent: 'center', marginTop: '10px'}}>
          <a href="https://github.com/Amirjab21/two-tower-search" target="_blank" rel="noopener noreferrer" className="link">code</a>
          <a href="https://huggingface.co/datasets/microsoft/ms_marco" target="_blank" rel="noopener noreferrer" className="link">dataset</a>
        </div>
      </div>
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
              <li style={{color: "black", gap: 18}} key={idx + document}>
                {document} <span className="similarity">({Math.round(distance)} distance)</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
