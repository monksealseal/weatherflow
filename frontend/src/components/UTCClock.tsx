import { useState, useEffect } from 'react';

export default function UTCClock() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const utc = now.toLocaleTimeString('en-US', {
    timeZone: 'UTC',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  const local = now.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });

  return (
    <div className="utc-clock" title="Current time (UTC / Local)">
      <span className="utc-clock__utc">{utc}Z</span>
      <span className="utc-clock__local">{local} L</span>
    </div>
  );
}
