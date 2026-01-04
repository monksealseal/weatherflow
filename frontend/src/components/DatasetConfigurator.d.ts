import { DatasetConfig, ServerOptions } from '../api/types';
interface Props {
    options: ServerOptions | null;
    value: DatasetConfig | null;
    onChange: (config: DatasetConfig) => void;
}
declare function DatasetConfigurator({ options, value, onChange }: Props): JSX.Element;
export default DatasetConfigurator;
