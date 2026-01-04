import { ModelConfig } from '../api/types';
interface Props {
    value: ModelConfig;
    onChange: (config: ModelConfig) => void;
}
declare function ModelConfigurator({ value, onChange }: Props): JSX.Element;
export default ModelConfigurator;
