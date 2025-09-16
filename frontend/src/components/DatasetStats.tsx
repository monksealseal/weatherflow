import { ChannelStats } from '../api/types';

interface Props {
  stats: ChannelStats[];
}

function DatasetStats({ stats }: Props): JSX.Element {
  return (
    <section className="section-card">
      <h2>Dataset statistics</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Channel</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Min</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((stat) => (
              <tr key={stat.name}>
                <td>{stat.name.toUpperCase()}</td>
                <td>{stat.mean.toFixed(3)}</td>
                <td>{stat.std.toFixed(3)}</td>
                <td>{stat.min.toFixed(3)}</td>
                <td>{stat.max.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default DatasetStats;
