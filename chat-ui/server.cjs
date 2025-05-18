const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Serve static files from the React build directory
app.use(express.static(path.join(__dirname, 'dist')));

// API endpoint to process queries
app.post('/api/process-query', async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }

    // Path to the Python script
    const pythonScriptPath = path.join(__dirname, '..', 'chat-agent', 'chat_runnner.py');
    
    // Spawn a Python process to execute the script
    const pythonProcess = spawn('python', [pythonScriptPath, query]);
    
    let dataString = '';
    let errorString = '';

    // Collect data from stdout
    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    // Collect errors from stderr
    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error: ${errorString}`);
        return res.status(500).json({
          success: false,
          error: errorString || 'An error occurred while processing your query'
        });
      }

      try {
        // Parse the output to extract results
        // This is a simple example - you might need to adjust based on your Python script's output
        const outputLines = dataString.split('\n');
        
        // Find SQL query
        const sqlQueryLine = outputLines.findIndex(line => line.startsWith('SQL Query:'));
        let sqlQuery = '';
        if (sqlQueryLine >= 0) {
          sqlQuery = outputLines.slice(sqlQueryLine + 1, outputLines.findIndex((line, i) => i > sqlQueryLine && line.trim() === '')).join('\n');
        }
        
        // Find results
        const resultsLine = outputLines.findIndex(line => line.startsWith('Results:'));
        let resultsString = '';
        if (resultsLine >= 0) {
          resultsString = outputLines.slice(resultsLine + 1, outputLines.findIndex((line, i) => i > resultsLine && line.trim() === '')).join('\n');
        }
        
        // Find explanation
        const explanationLine = outputLines.findIndex(line => line.startsWith('Explanation:'));
        let explanation = '';
        if (explanationLine >= 0) {
          explanation = outputLines.slice(explanationLine + 1).join('\n');
        }
        
        // Parse results into structured data
        // This is a simplified example - actual parsing depends on your output format
        const results = {};
        try {
          if (resultsString) {
            const rows = resultsString.trim().split('\n');
            if (rows.length > 0) {
              const headers = rows[0].split(',').map(h => h.trim());
              headers.forEach(header => {
                results[header] = [];
              });
              
              for (let i = 1; i < rows.length; i++) {
                const values = rows[i].split(',').map(v => v.trim());
                headers.forEach((header, idx) => {
                  // Try to convert to number if possible
                  const value = !isNaN(values[idx]) ? parseFloat(values[idx]) : values[idx];
                  results[header].push(value);
                });
              }
            }
          }
        } catch (parseError) {
          console.error('Error parsing results:', parseError);
        }

        return res.json({
          success: true,
          sql_query: sqlQuery,
          results: results,
          explanation: explanation
        });
        
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        return res.status(500).json({
          success: false,
          error: 'Failed to parse the results'
        });
      }
    });
    
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({
      success: false,
      error: 'Server error occurred'
    });
  }
});

// Fallback route for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});