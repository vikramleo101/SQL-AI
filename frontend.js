import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Container, Typography, TextField, Button, CircularProgress, Snackbar, Alert, Paper, Box, IconButton } from '@mui/material';
import { Upload as UploadIcon, Settings as SettingsIcon, Analytics as AnalyticsIcon, Insights as InsightsIcon } from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Theme setup
const theme = createTheme({
  palette: {
    primary: {
      main: '#4a148c',
    },
    secondary: {
      main: '#ffab00',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Container maxWidth="lg" sx={{ my: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
            <AnalyticsIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
            <Typography variant="h4" component="h1" color="primary">
              DataStory <span style={{ color: '#ffab00' }}>AI</span>
            </Typography>
          </Box>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analysis />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedFileId, setUploadedFileId] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/x-sqlite3'];
      if (validTypes.includes(selectedFile.type)) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please upload a valid CSV, Excel, or SQLite file');
      }
    }
  };

  const handleUpload = async () => {
    if (file) {
      setLoading(true);
      setError(null); // Clear previous errors
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (response.ok) {
          setUploadedFileId(data.file_id);
          // Simulate upload and analysis
          setTimeout(() => {
            setLoading(false);
            window.location.href = `/analyze?file_id=${data.file_id}`;
          }, 1500);
        } else {
          setError(data.error || 'File upload failed.');
          setLoading(false);
        }
      } catch (error) {
        setError('Network error or server unavailable.');
        setLoading(false);
      }
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Upload Your Data
      </Typography>
      <Typography variant="body1" color="textSecondary" gutterBottom>
        Start your analysis by uploading CSV, Excel, or SQLite files
      </Typography>
      
      <Box sx={{ my: 3, width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Button
          variant="contained"
          component="label"
          startIcon={<UploadIcon />}
          sx={{ mb: 2 }}
        >
          Select File
          <input type="file" hidden onChange={handleFileChange} accept=".csv,.xlsx,.xls,.db" />
        </Button>
        {file && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            Selected: {file.name}
          </Typography>
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Box>
      
      <Button
        variant="contained"
        color="secondary"
        onClick={handleUpload}
        disabled={!file || loading}
        endIcon={loading ? <CircularProgress size={20} /> : null}
      >
        {loading ? 'Analyzing...' : 'Start Analysis'}
      </Button>
      
      <Typography variant="caption" color="textSecondary" sx={{ mt: 3 }}>
        No data? Try our <Link to="/analyze">sample analysis</Link>
      </Typography>
    </Paper>
  );
}

function Analysis() {
  // This would be replaced with actual data and visualizations
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('visualization');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');

  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const fileId = queryParams.get('file_id');

  // Sample data for demo - remove or replace with actual fetched data
  const sampleData = [
    { month: 'Jan', revenue: 5000, expenses: 3000, profit: 2000 },
    { month: 'Feb', revenue: 7000, expenses: 3500, profit: 3500 },
    { month: 'Mar', revenue: 9000, expenses: 4500, profit: 4500 },
    { month: 'Apr', revenue: 6000, expenses: 4000, profit: 2000 },
    { month: 'May', revenue: 8000, expenses: 5000, profit: 3000 },
  ];

  const handleQuerySubmit = async () => {
    if (!query) {
      setSnackbarMessage('Please enter a query.');
      setSnackbarSeverity('warning');
      setSnackbarOpen(true);
      return;
    }

    if (!fileId) {
      setSnackbarMessage('No file uploaded. Please go back to Home and upload a file.');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      return;
    }

    setIsLoading(true);
    setResponse(null); // Clear previous response

    try {
      const apiResponse = await fetch('/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query, file_id: fileId }),
      });

      const data = await apiResponse.json();

      if (apiResponse.ok) {
        setResponse(data);
        setSnackbarMessage(`Analysis complete with ${(data.quality * 100).toFixed(0)}% confidence score`);
        setSnackbarSeverity('success');
      } else {
        setSnackbarMessage(data.error || 'Analysis failed.');
        setSnackbarSeverity('error');
      }
    } catch (error) {
      setSnackbarMessage('Network error or server unavailable.');
      setSnackbarSeverity('error');
    }
    setIsLoading(false);
    setSnackbarOpen(true);
  };

  return (
    <div>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5">Data Analysis Dashboard</Typography>
        <IconButton onClick={() => window.location.href='/settings'}>
          <SettingsIcon />
        </IconButton>
      </Box>
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          label="Ask anything about your data"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="E.g., What's the trend in profits? Compare revenue by month. Show me outliers."
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleQuerySubmit}
          disabled={!query || isLoading}
          endIcon={isLoading ? <CircularProgress size={20} /> : <InsightsIcon />}
        >
          {isLoading ? 'Analyzing...' : 'Get Insights'}
        </Button>
      </Paper>
      
      {response && (
        <div>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Button 
              onClick={() => setActiveTab('visualization')}
              variant={activeTab === 'visualization' ? 'contained' : 'text'}
            >
              Visualizations
            </Button>
            <Button 
              onClick={() => setActiveTab('insights')}
              variant={activeTab === 'insights' ? 'contained' : 'text'}
              sx={{ ml: 2 }}
            >
              Insights
            </Button>
            <Button 
              onClick={() => setActiveTab('data')}
              variant={activeTab === 'data' ? 'contained' : 'text'}
              sx={{ ml: 2 }}
            >
              Raw Data
            </Button>
          </Box>
          
          {activeTab === 'visualization' && (
            <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>Revenue Trend</Typography>
              <Box sx={{ height: 300, bgcolor: '#f5f5f5', p: 2 }}>
                {/* This would be a real chart in production */}
                {response && response.charts && response.charts.length > 0 ? (
                  <Typography align="center" color="textSecondary">
                    Recommended chart types: {response.charts.join(', ')}
                  </Typography>
                ) : (
                  <Typography align="center" color="textSecondary">
                    No chart recommendations.
                  </Typography>
                )}
                
                <Typography align="center" color="textSecondary">
                  Line chart visualization would appear here showing:
                </Typography>
                <ul>
                  {(response && response.tables || sampleData).map(item => (
                    <li key={item.month}>{item.month}: ${item.revenue}</li>
                  ))}
                </ul>
              </Box>
            </Paper>
          )}
          
          {activeTab === 'insights' && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Key Insights</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography sx={{ mr: 1 }}>Analysis Quality:</Typography>
                <Typography color={(response && response.quality > 0.9) ? 'success.main' : 'warning.main'}>
                  {(response && response.quality * 100).toFixed(0)}%
                </Typography>
              </Box>
              <Typography paragraph>{response && response.answer}</Typography>
              <ul>
                {(response && response.insights || []).map((insight, i) => (
                  <li key={i}>{insight}</li>
                ))}
              </ul>
            </Paper>
          )}
          
          {activeTab === 'data' && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Underlying Data</Typography>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f5f5f5' }}>
                      <th style={{ padding: '8px', textAlign: 'left' }}>Month</th>
                      <th style={{ padding: '8px', textAlign: 'left' }}>Revenue</th>
                      <th style={{ padding: '8px', textAlign: 'left' }}>Expenses</th>
                      <th style={{ padding: '8px', textAlign: 'left' }}>Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(response && response.tables || []).map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #ddd' }}>
                        <td style={{ padding: '8px' }}>{row.month}</td>
                        <td style={{ padding: '8px' }}>${row.revenue}</td>
                        <td style={{ padding: '8px' }}>${row.expenses}</td>
                        <td style={{ padding: '8px' }}>${row.profit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Box>
            </Paper>
          )}
        </div>
      )}
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setSnackbarOpen(false)} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </div>
  );
}

function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');

  const handleSave = async () => {
    try {
      const response = await fetch('/set_api_key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey }),
      });

      const data = await response.json();
      if (response.ok) {
        setSaveSuccess(true);
        setSnackbarMessage(data.message);
        setSnackbarSeverity('success');
      } else {
        setSnackbarMessage(data.error || 'Failed to save API key.');
        setSnackbarSeverity('error');
      }
    } catch (error) {
      setSnackbarMessage('Network error or server unavailable.');
      setSnackbarSeverity('error');
    }
    setSnackbarOpen(true);
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Settings
      </Typography>
      
      <TextField
        fullWidth
        variant="outlined"
        label="Google Gemini API Key"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        type="password"
        sx={{ mb: 2 }}
        helperText="Get your API key from Google AI Studio"
      />
      
      <Button
        variant="contained"
        color="primary"
        onClick={handleSave}
        disabled={!apiKey}
      >
        Save Settings
      </Button>
      
      {saveSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Settings saved successfully!
        </Alert>
      )}
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setSnackbarOpen(false)} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
      
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Data Quality Settings
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Configure how stringently DataStory AI evaluates your data quality
        </Typography>
        {/* Additional settings would go here */}
      </Box>
    </Paper>
  );
}

export default App;
