import { useEffect, useState } from 'react';
import { fetchTCReports } from '../../api/nhcClient';
import { CycloneClassification, TropicalCycloneReport } from '../../api/nhcTypes';
import './NHCReportsView.css';

export default function NHCReportsView(): JSX.Element {
  const [year, setYear] = useState(2024);
  const [reports, setReports] = useState<TropicalCycloneReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    setLoading(true);
    fetchTCReports(year).then(data => { setReports(data); setLoading(false); });
  }, [year]);

  const filtered = search
    ? reports.filter(r => r.stormName.toLowerCase().includes(search.toLowerCase()))
    : reports;

  const totalStorms = reports.length;
  const hurricanes = reports.filter(r => r.classification === 'hurricane' || r.classification === 'major_hurricane').length;
  const majors = reports.filter(r => r.classification === 'major_hurricane').length;
  const totalDeaths = reports.reduce((sum, r) => sum + r.deaths, 0);

  return (
    <div className="view-container nhc-reports">
      <div className="view-header">
        <h1>Tropical Cyclone Reports</h1>
        <p className="view-subtitle">
          Post-season TCR archive with synoptic history, statistics, and damage assessments
        </p>
      </div>

      <div className="controls-bar">
        <span className="control-label">Season:</span>
        <select className="control-select" value={year} onChange={e => setYear(Number(e.target.value))}>
          {Array.from({ length: 10 }, (_, i) => 2024 - i).map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
        <input
          className="control-search"
          placeholder="Search storms..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {loading ? (
        <div className="no-reports">Loading tropical cyclone reports...</div>
      ) : (
        <div className="reports-content">
          <div className="season-summary">
            <div className="summary-card">
              <div className="summary-value">{totalStorms}</div>
              <div className="summary-label">Named Storms</div>
            </div>
            <div className="summary-card">
              <div className="summary-value">{hurricanes}</div>
              <div className="summary-label">Hurricanes</div>
            </div>
            <div className="summary-card">
              <div className="summary-value">{majors}</div>
              <div className="summary-label">Major Hurricanes</div>
            </div>
            <div className="summary-card">
              <div className="summary-value">{totalDeaths}</div>
              <div className="summary-label">Fatalities</div>
            </div>
          </div>

          {filtered.length === 0 ? (
            <div className="no-reports">No reports found{search ? ` matching "${search}"` : ''}.</div>
          ) : (
            <div className="reports-list">
              {filtered.map(report => {
                const isExpanded = expanded === report.stormId;
                return (
                  <div key={report.stormId} className="report-card">
                    <div className="report-card-header" onClick={() => setExpanded(isExpanded ? null : report.stormId)}>
                      <div className="report-name-area">
                        <span className={`expand-icon ${isExpanded ? 'open' : ''}`}>{'\u25B6'}</span>
                        <span className="report-name">{report.stormName}</span>
                        <span className={`report-category ${catClass(report.classification)}`}>
                          {formatClassification(report.classification, report.peakWindKt)}
                        </span>
                      </div>
                      <span className="report-dates">
                        {report.formationDate} to {report.dissipationDate}
                      </span>
                    </div>
                    {isExpanded && (
                      <div className="report-details">
                        <div className="report-stats">
                          <div className="report-stat">
                            <span className="report-stat-label">Peak Wind</span>
                            <span className="report-stat-value">{report.peakWindKt} kt</span>
                          </div>
                          <div className="report-stat">
                            <span className="report-stat-label">Min Pressure</span>
                            <span className="report-stat-value">{report.minPressureMb} mb</span>
                          </div>
                          <div className="report-stat">
                            <span className="report-stat-label">Deaths</span>
                            <span className="report-stat-value">{report.deaths}</span>
                          </div>
                          <div className="report-stat">
                            <span className="report-stat-label">Damage</span>
                            <span className="report-stat-value">{report.damageUsd}</span>
                          </div>
                          <div className="report-stat">
                            <span className="report-stat-label">Basin</span>
                            <span className="report-stat-value">{report.basin.replace(/_/g, ' ')}</span>
                          </div>
                          <div className="report-stat">
                            <span className="report-stat-label">Storm ID</span>
                            <span className="report-stat-value">{report.stormId}</span>
                          </div>
                        </div>
                        <div className="report-summary">{report.summary}</div>
                        <a
                          className="report-link"
                          href={report.reportUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View Full TCR (PDF)
                        </a>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function catClass(classification: CycloneClassification): string {
  switch (classification) {
    case 'tropical_depression':
    case 'subtropical_depression': return 'td';
    case 'tropical_storm':
    case 'subtropical_storm': return 'ts';
    case 'hurricane': return 'hurricane';
    case 'major_hurricane': return 'major';
    default: return 'ts';
  }
}

function formatClassification(classification: CycloneClassification, windKt: number): string {
  switch (classification) {
    case 'tropical_depression': return 'TD';
    case 'tropical_storm': return 'TS';
    case 'hurricane': return `Cat ${saffirSimpson(windKt)}`;
    case 'major_hurricane': return `Cat ${saffirSimpson(windKt)}`;
    default: return classification.replace(/_/g, ' ');
  }
}

function saffirSimpson(kt: number): number {
  if (kt >= 137) return 5;
  if (kt >= 113) return 4;
  if (kt >= 96) return 3;
  if (kt >= 83) return 2;
  return 1;
}
