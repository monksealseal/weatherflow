import { ServerOptions, TrainingConfig } from '../api/types';
interface Props {
    options: ServerOptions | null;
    value: TrainingConfig | null;
    onChange: (config: TrainingConfig) => void;
}
declare function TrainingConfigurator({ options, value, onChange }: Props): JSX.Element;
export default TrainingConfigurator;
